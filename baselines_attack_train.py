import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
import librosa
from random import shuffle
import math
import time
from numpy import genfromtxt
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torchsummary import summary
import soundfile as sf

import os, glob
pd.set_option('display.max_rows', 500)
import h5py
import pickle
from sklearn import preprocessing
import argparse
import logging
from sklearn.preprocessing import label_binarize
from statistics import mean, variance, median
from collections import Counter
import config

torch.backends.cudnn.enabled=False

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify,
                       CWLoss_iemocap, get_model_iemocap, normalize_function,
                       CrossEntropyLoss, get_lr, set_lr, set_cyclic_lr)

# generator
sys.path.insert(1, os.path.join(sys.path[0], './audio_models/waveunet'))
# from seanet import GeneratorSEANt
# from gnet import AudioGeneratorResnet
from waveunet import Waveunet
# audio pre-processing
from transformers import Wav2Vec2FeatureExtractor, AutoConfig, PretrainedConfig
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
# hugging models
from transformers import AutoModelForAudioClassification, HubertForSequenceClassification, WavLMForSequenceClassification
# end2you audio models
from audio_rnn_model import AudioRNNModel


# Other attacks
import torchattacks
from onepixel import PixelAttacker

# For pytorch dataset
from torch.utils.data import TensorDataset, DataLoader
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

metric = load_metric("recall")
batch_size = config.batch_size
class_num = config.iemocap_num_classes
audio_len = config.iemocap_audio_samples
models_menu = config.iemocap_models_menu
criterion = CWLoss_iemocap
'''
class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)
'''
def data_generater(hdf5_path, validation):
    '''Read data into a dict'''
    with h5py.File(hdf5_path, 'r') as hf:
        x_train = hf['train_audio'][:]
        y_train = hf['train_y'][:]
        x_val = hf['dev_audio'][:]
        y_val = hf['dev_y'][:]
        x_test = hf['test_audio'][:]
        y_test = hf['test_y'][:]

    hf.close()

    if validation:
        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_val,'audio':x_val})}
    else:
        x_train = np.concatenate((x_train, x_val), axis=0)
        y_train = np.concatenate((y_train, y_val), axis=0)
        d = {'train':Dataset.from_dict({'label':y_train,'audio':x_train}), 'test':Dataset.from_dict({'label':y_test,'audio':x_test})}
    return d


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    cuda = args.cuda
    source_model_name = args.source_model_name
    rnn_name = args.rnn_name
    target = args.target
    eps = args.eps
    attack_type = args.attack_type
    pixel_counts = args.pixel_counts

    # data
    hdf5_path = os.path.join(workspace, "audio_4class_iemocap.h5")
    data = data_generater(hdf5_path, validation)
    dataset = DatasetDict(data)
    # dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # unify data
    logging.info('Data unifying')

    # dataset_train = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['train']['audio']]), torch.LongTensor(dataset['train']['label']))
    # trainloader_eval = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['test']['audio']]), torch.LongTensor(dataset['test']['label']))
    testloader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    del data
    del dataset

    '''
    if validation:
        testloader = trainloader_eval
    else:
        testloader = testloader
    '''
    test_size = len(testloader.dataset)

    logging.info('there are in total of {} samples for evaluation'.format(test_size))

    #  models loading
    source_model, source_model_path = get_model_iemocap(source_model_name, validation, workspace)
    if 'finetune' not in source_model_path:
        source_model = AudioRNNModel(input_size=audio_len, num_outs=class_num, model_name=source_model_name, rnn_name=rnn_name)
    source_model.load_state_dict(torch.load(source_model_path))
    logging.info('the source eval model is {} and located in {}'.format(source_model_name, source_model_path))

    if cuda:
        source_model.cuda()
    source_model.eval()

    # attacker
    if attack_type == 'pgd':
        attack = torchattacks.PGD(source_model, eps=eps, alpha=eps/4, steps=20, random_start=True, pixel_counts=pixel_counts, targeted_attack=target!=-1, model_type=source_model_name not in ['emo18', 'zhao19'])
    # untargeted only
    if attack_type == 'sparsefool':
        attack = torchattacks.SparseFool(source_model, steps=20, lam=10, overshoot=0.02, eps=eps)
    elif attack_type == 'onepixel':
        attacker = PixelAttacker(source_model, dimensions=audio_len, pixel_counts=pixel_counts, eps=eps, targeted_attack=target!=-1, model_type=source_model_name not in ['emo18', 'zhao19'])

    # some statics
    x_org = []
    x_adv = []
    y = []
    time_count = 0
    hdf5_adv = os.path.join(workspace, 'sparse_attack', 'sparse_adv', attack_type, 'iemocap_source_{}_val_{}_pixel_{}.h5'.format(source_model_name, validation, pixel_counts))

    for (idx, (batch_x, batch_y)) in enumerate(testloader, 0):
        if idx % 500 == 0:
            logging.info('Processed {}-th samples'.format(idx))
        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        if source_model_name not in ['emo18', 'zhao19']:
            batch_output = source_model(normalize_function(batch_x.clone().detach()))[0]
        else:
            batch_output = source_model(normalize_function(batch_x.clone().detach()))

        if target == -1:
            label = torch.argmax(batch_output, dim=-1)
        else:
            label = torch.LongTensor(batch_y.size(0))
            label.fill_(target)
            label = move_data_to_gpu(label, cuda)

        # adversary
        start = time.time()
        if attack_type == 'onepixel':
            adv = attacker.attack(batch_x, label)
            end = time.time()
            adv = np.squeeze(adv, 0)
            x_adv.append(adv)
        else:
            adv = attack(batch_x, label)
            end = time.time()
            adv = torch.squeeze(adv, 0)
            x_adv.append(adv.detach().cpu().numpy())
        time_count += end - start

        batch_x = torch.squeeze(batch_x, 0)
        x_org.append(batch_x.detach().cpu().numpy())
        y.append(batch_y.detach().cpu().numpy())

    # generate the hdf5 file for evaluation
    # x_eval = np.array(x_eval, dtype='float32')
    # y_eval = np.array(y_eval, dtype='int32')
    with h5py.File(hdf5_adv, 'w') as hf:
        hf.create_dataset("x_org",  data=x_org, dtype='float32')
        hf.create_dataset("x_adv",  data=x_adv, dtype='float32')
        hf.create_dataset("y", data=y, dtype='int32')

    hf.close()

    # forward_evaluate(target_model, attack, testloader, source_model_name, target_model_name, eps, cuda, target_model_name not in ['emo18', 'zhao19'])

    logging.info('source model:{}, attack: {}, pixels/sample: {}/{}, avg time: {:.4f}'.format(source_model_name, attack_type, pixel_counts, test_size, time_count/test_size))
    logging.info('adv samples are stored in {}'.format(hdf5_adv))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/IEMOCAP')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--eps', type=float, default=0.01, help='perturbation budget')
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--source_model_name', type=str, choices=['wav2vec2', 'hubert', 'wavlm', 'zhao19', 'emo18'], required=True)
    parser_train.add_argument('--rnn_name', type=str, default='lstm', choices=['gru', 'lstm'])
    parser_train.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
    parser_train.add_argument('--attack_type', type=str, choices=['sparsefool', 'onepixel', 'pgd'], required=True)
    parser_train.add_argument('--pixel_counts', type=int, choices=[1, 6500, 26000, 35000], required=True)
    args = parser.parse_args()

    args.filename = get_filename(__file__)


    # Create log
    logs_dir = os.path.join(args.workspace, 'sparse_attack', args.filename, '{}'.format(args.attack_type))
    create_folder(logs_dir)
    custom = 'val_{}_pixel_{}'.format(args.validation, args.pixel_counts)
    create_logging(logs_dir, custom, filemode='w')
    logging.info(args)


    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')

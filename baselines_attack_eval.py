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
def data_generater(hdf5_path):
    '''Read data into a dict'''
    with h5py.File(hdf5_path, 'r') as hf:
        x_org = hf['x_org'][:]
        x_adv = hf['x_adv'][:]
        y = hf['y'][:]

    hf.close()

    # d = Dataset.from_dict({'label': y, 'x_org': x_org, 'x_adv': x_adv})
    d = {'y': y, 'x_org': x_org, 'x_adv': x_adv}
    return d


def forward_evaluate(model_t, data_loader, source_model_name, target_model_name, eps, cuda, z):
    # z is True if model_t is not zhao19 or emo18
    adv_acc = 0
    clean_acc = 0
    fool_rate = 0
    target_rate = 0
    norm = 0
    distortion = 0
    save_five = 0

    test_size = len(data_loader.dataset)
    assert data_loader.batch_size == 1

    for (idx, (batch_x_org, batch_x_adv, batch_y)) in enumerate(data_loader, 0):

        batch_x_org = move_data_to_gpu(batch_x_org, cuda)
        batch_x_adv = move_data_to_gpu(batch_x_adv, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model_t.eval()
        if z:
            clean_out = model_t(normalize_function(batch_x_org.clone().detach()))[0]
        else:
            clean_out = model_t(normalize_function(batch_x_org.clone().detach()))

        clean_acc += torch.sum(clean_out.argmax(dim=-1) == batch_y).item()


        if z:
            adv_out = model_t(normalize_function(batch_x_adv.clone().detach()))[0]
        else:
            adv_out = model_t(normalize_function(batch_x_adv.clone().detach()))

        adv_acc +=torch.sum(adv_out.argmax(dim=-1) == batch_y).item()

        fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

        if args.target != -1:
            target = torch.LongTensor(batch_y.size(0))
            target.fill_(args.target)
            target = target.cuda()
            target_rate += torch.sum(adv_out.argmax(dim=-1) == target).item()

        # calculate the distortion
        # only take the success ones into SNR and norm; only save successful adversarial audios
        if torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item() > 0:
            per = torch.squeeze(batch_x_adv.clone().detach() - batch_x_org.clone().detach())
            i_dis = torch.log10(torch.max(torch.abs(batch_x_org), dim=-1).values / torch.max(torch.abs(per), dim=-1).values)
            i_dis_no_inf = torch.where(i_dis == float('inf'), torch.tensor(0.0).cuda().detach(), i_dis.detach())
            distortion += torch.sum(20 * i_dis_no_inf).item()
            norm += torch.norm(per, 0)

            # Save some audio samples for human evaluation
            if save_five < 10 and test_size == 1151:
                cur_dis = torch.sum(20 * i_dis_no_inf).item() / 1
                cur_norm = torch.norm(per.detach(), 0) / 1
                audio_dir = os.path.join('/home/ychang/sparse_attack/workspace/audios_listen/IEMOCAP/baselines', '{}_{}'.format(source_model_name, target_model_name))
                create_folder(audio_dir)
                audio_org = os.path.join(audio_dir, '{:.4f}_{:.4f}_org.wav'.format(cur_dis, cur_norm))
                audio_adv = os.path.join(audio_dir, '{:.4f}_{:.4f}_adv.wav'.format(cur_dis, cur_norm))
                audio_per = os.path.join(audio_dir, '{:.4f}_{:.4f}_per.wav'.format(cur_dis, cur_norm))

                for i in range(1):
                    if batch_y[i] != torch.argmax(adv_out[i], dim=-1):
                        sf.write(audio_org, batch_x_org[i].clone().detach().cpu().numpy(), 16000)
                        sf.write(audio_adv, batch_x_adv[i].clone().detach().cpu().numpy(), 16000)
                        per = np.squeeze(batch_x_adv[i].clone().detach().cpu().numpy() - batch_x_org[i].clone().detach().cpu().numpy())
                        sf.write(audio_per, per, 16000)
                        save_five += 1
                        break
    if fool_rate != 0:
        logging.info('L0 norm: {:.4f}, distortion: {:.4f}'.format(norm/fool_rate, distortion/fool_rate))
    if args.target != -1:
        logging.info('Clean: {0:.4%} Adversarial: {1:.4%} Fooling Rate: {2:.4%} Target Success Rate:{3:.4%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size, target_rate/test_size))
    else:
        # logging.info('Clean: {0:.3%} Adversarial: {1:.3%} Fooling Rate:{2:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size))
        logging.info('fool rate: {:.4%}'.format(fool_rate/test_size))



def train(args):
    workspace = args.workspace
    validation = args.validation
    cuda = args.cuda
    source_model_name = args.source_model_name
    target_model_name = args.target_model_name
    rnn_name = args.rnn_name
    target = args.target
    eps = args.eps
    attack_type = args.attack_type
    pixel_counts = args.pixel_counts

    # adv samples preparation
    hdf5_adv = os.path.join(workspace, 'sparse_attack', 'sparse_adv', attack_type, 'iemocap_source_{}_val_{}_pixel_{}.h5'.format(source_model_name, validation, pixel_counts))
    data = data_generater(hdf5_adv)
    dataset = DatasetDict(data)
    # dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)
    dataset_eval = TensorDataset(torch.Tensor(dataset['x_org']), torch.Tensor(dataset['x_adv']), torch.LongTensor(dataset['y']))
    dataloader_eval = DataLoader(dataset_eval, batch_size=1, shuffle=False, num_workers=2)

    del data
    del dataset

    logging.info('{} adv samples loaded from {}'.format(len(dataloader_eval.dataset), hdf5_adv))

    #  models loading
    target_model, target_model_path = get_model_iemocap(target_model_name, validation, workspace)
    if 'finetune' not in target_model_path:
        target_model = AudioRNNModel(input_size=audio_len, num_outs=class_num, model_name=target_model_name, rnn_name=rnn_name)

    target_model.load_state_dict(torch.load(target_model_path))

    logging.info('the target eval model is {} and located in {}'.format(target_model_name, target_model_path))

    if cuda:
        target_model.cuda()
    target_model.eval()

    forward_evaluate(target_model, dataloader_eval, source_model_name, target_model_name, eps, cuda, target_model_name not in ['emo18', 'zhao19'])

    logging.info('finished evaluation for {}'.format(target_model_name))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--workspace', type=str, default='/storage/home/ychang/IEMOCAP')
    parser_eval.add_argument('--validation', action='store_true', default=False)
    parser_eval.add_argument('--eps', type=float, default=0.01, help='perturbation budget')
    parser_eval.add_argument('--cuda', action='store_true', default=False)
    parser_eval.add_argument('--source_model_name', type=str, choices=['wav2vec2', 'hubert', 'wavlm', 'zhao19', 'emo18'], required=True)
    parser_eval.add_argument('--target_model_name', type=str, choices=['wav2vec2', 'hubert', 'wavlm', 'zhao19', 'emo18'], required=True)
    parser_eval.add_argument('--rnn_name', type=str, default='lstm', choices=['gru', 'lstm'])
    parser_eval.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
    parser_eval.add_argument('--attack_type', type=str, choices=['sparsefool', 'onepixel', 'pgd'], required=True)
    parser_eval.add_argument('--pixel_counts', type=int, choices=[1, 6500, 26000, 35000], required=True)
    args = parser.parse_args()

    args.filename = get_filename(__file__)


    # Create log
    logs_dir = os.path.join(args.workspace, 'sparse_attack', args.filename, args.attack_type)
    create_folder(logs_dir)
    custom = 'val_{}_pixel_{}_{}_{}'.format(args.validation, args.pixel_counts, args.source_model_name, args.target_model_name)
    create_logging(logs_dir, custom, filemode='w')
    logging.info(args)


    if args.mode == 'eval':
        train(args)
    else:
        raise Exception('Error argument!')

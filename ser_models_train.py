import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
import librosa
from random import shuffle
import math
from numpy import genfromtxt
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.autograd import Variable

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

sys.path.insert(1, os.path.join(sys.path[0], './utils'))
from utilities import (read_audio, create_folder,
                       get_filename, create_logging, calculate_accuracy,
                       print_accuracy, calculate_confusion_matrix,
                       move_data_to_gpu, audio_unify, normalize_function, get_lr)
# wav2vec related
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor
from transformers import AutoModelForAudioClassification, TrainingArguments, Trainer, Wav2Vec2ForPreTraining
from transformers import get_scheduler
# WavLM related
from transformers import WavLMForSequenceClassification, PretrainedConfig
# CNN models related
sys.path.insert(1, os.path.join(sys.path[0], './audio_models'))
from audio_rnn_model import AudioRNNModel

# For pytorch dataset
from torch.utils.data import TensorDataset, DataLoader
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

batch_size = config.batch_size
class_num = config.iemocap_num_classes
audio_len = config.iemocap_audio_samples

# Some model preparation
metric = load_metric("recall")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
# The wavlm model was pre-trained on 960h of Librispeech
wavlm_model = "microsoft/wavlm-base"
wavlm_config = PretrainedConfig.from_pretrained(wavlm_model, num_labels=class_num)


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


def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # labels = np.argmax(labels, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')


def evaluate_finetune(model, data_loader, cuda, large_model):

    outputs, targets = forward_finetune(model, data_loader, cuda, large_model)

    # loss
    loss_fct = nn.CrossEntropyLoss()
    loss = float(loss_fct(Variable(torch.Tensor(outputs)), Variable(torch.LongTensor(targets))).data.numpy())

    # UAR
    classes_num = outputs.shape[-1]
    predictions = np.argmax(outputs, axis=-1)
    acc, uar = calculate_accuracy(targets, predictions, classes_num)

    return loss, acc, uar


def forward_finetune(model, data_loader, cuda, large_model):

    outputs = []
    targets = []

    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):
        # normalization
        batch_x = normalize_function(batch_x)

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model.eval()
        # [0] to get the logits from SequenceClassifierOutput class
        batch_output = model(batch_x)
        if large_model:
            batch_output = batch_output[0]

        outputs.append(batch_output.data.cpu().numpy())
        targets.append(batch_y.data.cpu().numpy())

    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0)

    return outputs, targets

def train(args):
    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    cuda = args.cuda
    freeze = args.freeze
    model_name = args.model_name
    large_model = model_name in ['wav2vec2', 'wavlm']

    hdf5_path = os.path.join(workspace, "audio_4class_iemocap.h5")
    models_path_base = os.path.join(workspace, 'sparse_attack', 'trained_models')
    if large_model:
        if validation and freeze:
            models_dir = os.path.join(models_path_base, 'models_{}_finetune'.format(model_name), 'freeze', 'train_devel')

        elif not validation and freeze:
            models_dir = os.path.join(models_path_base, 'models_{}_finetune'.format(model_name), 'freeze', 'traindevel_test')

        elif validation and not freeze:
            models_dir = os.path.join(models_path_base, 'models_{}_finetune'.format(model_name), 'no_freeze', 'train_devel')

        elif not validation and not freeze:
            models_dir = os.path.join(models_path_base, 'models_{}_finetune'.format(model_name), 'no_freeze', 'traindevel_test')
    else:
        if validation:
            models_dir = os.path.join(models_path_base, 'models_{}'.format(model_name), 'train_devel')

        elif not validation:
            models_dir = os.path.join(models_path_base, 'models_{}'.format(model_name), 'traindevel_test')

    create_folder(models_dir)

    # data
    data = data_generater(hdf5_path, validation)
    dataset = DatasetDict(data)
    # dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # model loading
    if model_name == 'wav2vec2':
        model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=class_num)
        if freeze:
            model.freeze_feature_extractor()
    elif model_name == 'wavlm':
        model = WavLMForSequenceClassification.from_pretrained(
            wavlm_model,
            config=wavlm_config,  # because we need to update num_labels as per our dataset
            ignore_mismatched_sizes=True,  # to avoid classifier size mismatch from from_pretrained.
        )
        if freeze:
            model.freeze_feature_extractor()
    else:
        model = AudioRNNModel(input_size=audio_len, num_outs=class_num, model_name=model_name, rnn_name='lstm')

    # calculate the number of parameters
    logging.info(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info("Total Params: {}".format(total_params))
    if cuda:
        model.cuda()

    # unify data
    logging.info('Data unifying')
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['train']['audio']]), torch.LongTensor(dataset['train']['label']))
    trainloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['test']['audio']]), torch.LongTensor(dataset['test']['label']))
    testloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=2)

    del data
    del dataset

    # training
    logging.info('Start developing {}/{} for train/test'.format(len(trainloader.dataset), len(testloader.dataset)))
    if large_model:
        lr = 3e-5
    else:
        lr = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.)
    loss_fct = nn.CrossEntropyLoss()

    # Only save the best model at the end of training
    best_uar = 0
    best_epoch = 0
    previous_out_path = os.path.join(models_dir, '1111.pt')

    for epoch_idx in range(0, epoch):
        logging.info('epoch: {}, lr:{}'.format(epoch_idx, get_lr(optimizer)))
        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):
            # normalization
            batch_x = normalize_function(batch_x)
            # move to GPU
            batch_x = move_data_to_gpu(batch_x, cuda)
            batch_y = move_data_to_gpu(batch_y, cuda)

            model.train()
            batch_output = model(batch_x)
            if large_model:
                batch_output = batch_output[0]

            loss = loss_fct(batch_output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # evaluate
        tr_loss, tr_acc, tr_uar = evaluate_finetune(model, trainloader, cuda, large_model)
        te_loss, te_acc, te_uar = evaluate_finetune(model, testloader, cuda, large_model)
        logging.info('In Epoch: {}, train_acc: {:.3f}, train_uar: {:.3f}, train_loss: {:.3f}'.format(epoch_idx, tr_acc, tr_uar, tr_loss))
        logging.info('In Epoch: {}, test_acc:{:.3f}, test_uar: {:.3f}, test_loss: {:.3f}'.format(epoch_idx, te_acc, te_uar, te_loss))

        # save model
        if epoch_idx == epoch -1:
            save_out_path = os.path.join(models_dir, "{}_epoch_{:.4f}_{:.4f}.pt".format(epoch_idx, te_acc, te_uar))
            torch.save(model.state_dict(), save_out_path)
            logging.info('Model saved to {}'.format(save_out_path))
    logging.info('finished training')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/IEMOCAP')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=True)
    parser_train.add_argument('--freeze', action='store_true', default=True)
    parser_train.add_argument('--model_name', type=str, choices=['wav2vec2', 'wavlm', 'zhao19', 'emo18'], default='wav2vec2')


    args = parser.parse_args()

    args.filename = get_filename(__file__)

    # Create log
    logs_dir = os.path.join(args.workspace, 'sparse_attack', args.filename, 'logs')
    custom = '{}_{}_{}'.format(args.model_name, args.validation, args.epoch)
    create_logging(logs_dir, custom, filemode='w')
    logging.info(args)

    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')

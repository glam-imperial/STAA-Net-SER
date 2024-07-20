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


# For pytorch dataset
from datasets.dataset_dict import DatasetDict
from datasets import Dataset, load_metric

metric = load_metric("recall")
batch_size = config.batch_size
class_num = config.iemocap_num_classes
audio_len = config.iemocap_audio_samples
models_menu = config.iemocap_models_menu
criterion = CWLoss_iemocap

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

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


def forward_evaluate(epoch_idx, model_t, netG, data_loader, source_model_name, eps, cuda, z):
    # z is True if model_t is not zhao19 or emo18
    adv_acc = 0
    clean_acc = 0
    fool_rate = 0
    target_rate = 0
    norm = 0
    loss_inf = 0
    loss_qua = 0
    loss_sparse = 0
    distortion = 0
    time_count = 0
    save_five = 0

    # For loss calculation here
    outputs = []
    targets = []
    tar_targets = []

    test_size = len(data_loader.dataset)
    assert data_loader.batch_size == 1

    # if saving the adv samples
    save_adv = test_size != 3349 and epoch_idx == args.epoch-1
    if save_adv:
        x_org = []
        x_adv = []
        y = []
        hdf5_adv = os.path.join(args.workspace, 'sparse_attack', 'sparse_adv', 'generator', 'iemocap_source_{}_val_{}.h5'.format(source_model_name, args.validation))

    for (idx, (batch_x, batch_y)) in enumerate(data_loader, 0):

        batch_x = move_data_to_gpu(batch_x, cuda)
        batch_y = move_data_to_gpu(batch_y, cuda)

        model_t.eval()
        if z:
            clean_out = model_t(normalize_function(batch_x.clone().detach()))[0]
        else:
            clean_out = model_t(normalize_function(batch_x.clone().detach()))

        clean_acc += torch.sum(clean_out.argmax(dim=-1) == batch_y).item()


        # Adversary
        netG.eval()
        start = time.time()
        adv,adv_inf,adv_0,adv_00 = netG(batch_x, eps, None)
        end = time.time()
        adv = torch.squeeze(adv, 1)
        time_count += end - start

        if save_adv and epoch_idx == args.epoch -1:
            x_org.append(np.squeeze(batch_x.clone().detach().cpu().numpy(),0))
            y.append(batch_y.clone().detach().cpu().numpy())
            x_adv.append(np.squeeze(adv.clone().detach().cpu().numpy(),0))

        if z:
            adv_out = model_t(normalize_function(adv.clone().detach()))[0]
        else:
            adv_out = model_t(normalize_function(adv.clone().detach()))

        adv_acc +=torch.sum(adv_out.argmax(dim=-1) == batch_y).item()

        fool_rate += torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item()

        if args.target != -1:
            target = torch.LongTensor(batch_y.size(0))
            target.fill_(args.target)
            target = target.cuda()
            target_rate += torch.sum(adv_out.argmax(dim=-1) == target).item()
            tar_targets.append(target.detach())

        # In the evaluation mode, netG generate [0,1] adv_0, then sparsity is calculated by L_{0}
        # norm += torch.norm(adv_0.detach(), 0)
        loss_inf += args.qi * torch.norm(adv_inf.detach(), 2)
        loss_qua += args.ql * torch.norm((adv_0.detach() - adv_00.detach()), 2)
        loss_sparse += args.qs * torch.norm(adv_0.detach(), 1)

        targets.append(torch.argmax(clean_out.detach(), dim=-1))
        outputs.append(adv_out.detach())

        # calculate the distortion
        # only take the success ones into SNR and norm; only save successful adversarial audios
        if torch.sum(adv_out.argmax(dim=-1) != clean_out.argmax(dim=-1)).item() > 0:
            per = torch.squeeze(adv_inf.clone().detach() * adv_0.clone().detach(), dim=1)
            i_dis = torch.log10(torch.max(torch.abs(batch_x), dim=-1).values / torch.max(torch.abs(per), dim=-1).values)
            # items = torch.masked_select(i_dis, i_dis == float("inf") or i_dis == float('-inf'))
            i_dis_no_inf = torch.where(i_dis == float('inf'), torch.tensor(0.0).cuda().detach(), i_dis.detach())
            distortion += torch.sum(20 * i_dis_no_inf).item()
            norm += torch.norm(adv_0.detach(), 0)

            # Save some audio samples for human evaluation
            if save_five < 30 and epoch_idx == args.epoch -1 and test_size == 1151:
                cur_dis = torch.sum(20 * i_dis_no_inf).item() / 1
                cur_norm = torch.norm(adv_0.detach(), 0) / 1
                audio_dir = os.path.join('/home/ychang/sparse_attack/workspace/audios_listen/IEMOCAP/generator2', '{}'.format(source_model_name))
                create_folder(audio_dir)
                audio_org = os.path.join(audio_dir, '{:.4f}_{:.4f}_{:.4f}_org.wav'.format(cur_dis, cur_norm, eps + args.qi))
                audio_adv = os.path.join(audio_dir, '{:.4f}_{:.4f}_{:.4f}_adv.wav'.format(cur_dis, cur_norm, eps + args.qi))
                audio_per = os.path.join(audio_dir, '{:.4f}_{:.4f}_{:.4f}_per.wav'.format(cur_dis, cur_norm, eps + args.qi))

                for i in range(1):
                    if batch_y[i] != torch.argmax(adv_out[i], dim=-1):
                        sf.write(audio_org, batch_x[i].clone().detach().cpu().numpy(), 16000)
                        sf.write(audio_adv, adv[i].clone().detach().cpu().numpy(), 16000)
                        per = np.squeeze(adv_0[i].clone().detach().cpu().numpy() * adv_inf[i].clone().detach().cpu().numpy(),0)
                        sf.write(audio_per, per, 16000)
                        save_five += 1
                        break
    # Save the adv samples
    if save_adv:
        with h5py.File(hdf5_adv, 'w') as hf:
            hf.create_dataset("x_org",  data=x_org, dtype='float32')
            hf.create_dataset("x_adv",  data=x_adv, dtype='float32')
            hf.create_dataset("y", data=y, dtype='int32')

        hf.close()

    # loss function
    outputs = torch.cat(outputs, axis=0)
    targets = torch.cat(targets, axis=0)
    if args.target != -1:
        tar_targets = torch.cat(tar_targets, axis=0)
        loss_adv = criterion(outputs, tar_targets, tar=True)
    else:
        loss_adv = criterion(outputs, targets)

    # loss_adv = criterion(outputs, targets).data.cpu().numpy()

    loss_total = loss_adv + loss_inf + loss_qua + loss_sparse
    if len(data_loader.dataset) > 3300:
        logging.info("-----------------------------------")
    logging.info('total:{:.4f}, adv:{:.4f}, inf:{:.4f}, qua:{:.4f}, sparse:{:.4f}'.format(loss_total, loss_adv, loss_inf, loss_qua, loss_sparse))

    logging.info('L0 norm: {:.4f}, distortion: {:.4f}, time: {:.4f}'.format(norm/fool_rate, distortion/fool_rate, time_count/test_size))
    if args.target != -1:
        logging.info('Clean: {0:.3%} Adversarial: {1:.3%} Fooling Rate: {2:.3%} Target Success Rate:{3:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size, target_rate/test_size))
    else:
        # logging.info('Clean: {0:.3%} Adversarial: {1:.3%} Fooling Rate:{2:.3%}'.format(clean_acc/test_size, adv_acc/test_size, fool_rate/test_size))
        logging.info('fool rate: {:.3%}'.format(fool_rate/test_size))


def train(args):
    # Arugments & parameters
    workspace = args.workspace
    validation = args.validation
    epoch = args.epoch
    cuda = args.cuda
    source_model_name = args.source_model_name
    rnn_name = args.rnn_name
    target = args.target
    eps = args.eps
    ql = args.ql
    qi = args.qi
    qs = args.qs
    freeze_dec1 = args.freeze_dec1
    lr = args.lr
    step_size = args.step_size

    # data
    hdf5_path = os.path.join(workspace, "audio_4class_iemocap.h5")
    data = data_generater(hdf5_path, validation)
    dataset = DatasetDict(data)
    # dataset = dataset.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=batch_size)

    # unify data
    logging.info('Data unifying')
    dataset_train = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['train']['audio']]), torch.LongTensor(dataset['train']['label']))
    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader_eval = torch.utils.data.DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=2)

    dataset_test = TensorDataset(torch.Tensor([audio_unify(x, seq_len=int(audio_len)) for x in dataset['test']['audio']]), torch.LongTensor(dataset['test']['label']))
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=2)

    del data
    del dataset

    #  models loading
    source_model, source_model_path = get_model_iemocap(source_model_name, validation, workspace)
    if 'finetune' not in source_model_path:
        source_model = AudioRNNModel(input_size=audio_len, num_outs=class_num, model_name=source_model_name, rnn_name=rnn_name)

    source_model.load_state_dict(torch.load(source_model_path))
    logging.info('validation is {} and training samples {} and eval samples {}'.format(validation, len(trainloader.dataset), len(testloader.dataset)))
    logging.info('the source eval model is {} and located in {}'.format(source_model_name, source_model_path))

    if cuda:
        source_model.cuda()
    source_model.eval()

    # generator
    num_features = [args.features*2**i for i in range(0, args.levels)]
    netG = Waveunet(1, num_features, 1, args.instruments, kernel_size=5,
                     target_output_size=audio_len, depth=args.depth, strides=4,
                     conv_type=args.conv_type, res=args.res, separate=0, freeze_dec1=freeze_dec1)

    # logging.info(netG)
    if cuda:
        netG.cuda()
    total_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    logging.info("Total Params: {}".format(total_params))


    # training
    logging.info('Start training ...')
    # both papers use this initial lr
    lr = lr
    optimizer = optim.Adam(netG.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)

    lam_qua = ql
    lam_inf = qi
    lam_spa = qs

    save_five = 0

    for epoch_idx in range(0, epoch):
        logging.info('epoch: {}, lr:{}'.format(epoch_idx, get_lr(optimizer)))

        if epoch_idx == 9 and args.flex_eps:
            eps = 0.03
            logging.info('eps changed to: {}'.format(eps))
        if epoch_idx == 19 and args.flex_eps:
            eps = 0.01
            logging.info('eps changed to: {}'.format(eps))

        for (idx, (batch_x, batch_y)) in enumerate(trainloader, 0):

            netG.train()
            source_model.eval()

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

            adv, adv_inf, adv_0, adv_00 = netG(batch_x, eps, None)
            # Preprocessing just for the source model or target model, not for the netG
            adv = torch.squeeze(adv, 1)
            source_model.eval()
            if source_model_name not in ['emo18', 'zhao19']:
                adv_out = source_model(normalize_function(adv))[0]
            else:
                adv_out = source_model(normalize_function(adv))

            if target == -1:
                # Gradient accent (Untargetted Attack)
                loss_adv = criterion(adv_out, label)
            else:
                # Gradient decent (Targetted Attack)
                loss_adv = criterion(adv_out, label, tar=True)

            loss_inf = torch.norm(adv_inf, 2)
            loss_spa = torch.norm(adv_0, 1)
            bi_adv_00 = torch.where(adv_00 < 0.5, torch.zeros_like(adv_00), torch.ones_like(adv_00))
            loss_qua = torch.norm((bi_adv_00 - adv_00), 2)
            loss = loss_adv + lam_inf * loss_inf + lam_qua * loss_qua + lam_spa * loss_spa

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate
        forward_evaluate(epoch_idx, source_model, netG, trainloader_eval, source_model_name, eps, cuda, source_model_name not in ['emo18', 'zhao19'])
        forward_evaluate(epoch_idx, source_model, netG, testloader, source_model_name, eps, cuda, source_model_name not in ['emo18', 'zhao19'])

        # Update the learning rate
        scheduler.step()


        if epoch_idx == args.epoch - 1:
            torch.save(netG.state_dict(), os.path.join(args.workspace, 'sparse_attack', 'iemocap_generator_train', 'generator3', 'netG_{}_{}_{}.pth'.format(source_model_name, epoch_idx, args.validation)))

    logging.info('finished training')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, default='/storage/home/ychang/IEMOCAP')
    parser_train.add_argument('--validation', action='store_true', default=False)
    parser_train.add_argument('--epoch', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--source_model_name', type=str, choices=['wav2vec2', 'hubert', 'wavlm', 'zhao19', 'emo18'], required=True)
    parser_train.add_argument('--rnn_name', type=str, default='lstm', choices=['gru', 'lstm'])
    parser_train.add_argument('--target', type=int, default=-1, help='-1 if untargeted')
    parser_train.add_argument('--eps', type=float, default=0.03, help='perturbation budget')
    parser_train.add_argument('--ql', type=float, default=1e-6, help='quatilization_loss')
    parser_train.add_argument('--qi', type=float, default=0.0, help='inf_loss')
    parser_train.add_argument('--qs', type=float, default=0.1, help='sparse_loss')
    parser_train.add_argument('--lr', type=float, default=1e-3, help='initial_lr')
    parser_train.add_argument('--step_size', type=int, default=5, help='step_size')
    parser_train.add_argument('--flex_eps', action='store_true', default=False)
    parser_train.add_argument('--freeze_dec1', action='store_true', default=False)
    parser_train.add_argument('--instruments', type=str, nargs='+', default=["bass", "drums", "other", "vocals"])
    parser_train.add_argument('--depth', type=int, default=1,help="Number of convs per block")
    parser_train.add_argument('--features', type=int, default=32, help='Number of feature channels per layer')
    parser_train.add_argument('--levels', type=int, default=6, help="Number of DS/US blocks")
    parser_train.add_argument('--conv_type', type=str, default="gn",
                                help="Type of convolution (normal, BN-normalised, GN-normalised): normal/bn/gn")
    parser_train.add_argument('--res', type=str, default="learned",
                                help="Resampling strategy: fixed sinc-based lowpass filtering or learned conv layer: fixed/learned")
    args = parser.parse_args()

    args.filename = get_filename(__file__)


    # Create log
    logs_dir = os.path.join(args.workspace, 'sparse_attack', args.filename, "{}".format(args.source_model_name))
    create_folder(logs_dir)
    custom = "{}_{:.4f}_{}".format(args.eps, args.ql+args.qs+args.qi, args.epoch)
    create_logging(logs_dir, custom, filemode='w')
    logging.info(args)


    if args.mode == 'train':
        train(args)
    else:
        raise Exception('Error argument!')

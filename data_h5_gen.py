# This code is for IEMOCAP dataset

import os
import sys
import numpy as np
import pandas as pd
import argparse
import librosa
import logging
import math
from scipy import signal
import time
from utilities import read_audio_no_unify, create_folder, create_logging
import config
import soundfile
import h5py
import random
from statistics import mean, variance, median

emos = ['hap', 'sad', 'ang', 'neu', 'exc']
counts = [0] * 5
emo_div = {}
emo_gender = {}
divisions = ['train', 'val', 'test']
for d in divisions:
    emo_div[d] = dict(zip(emos, counts))
genders = ['F', 'M']
for g in genders:
    emo_gender[g] = dict(zip(emos, counts))


def emo_check(emo):
    ind = emos.index(emo)
    if ind == 4:
        ind = 0
    return ind

def iemocap_gen(args):
    workspace = args.workspace
    txt_path = os.path.join(workspace, 'stats.txt')
    hdf5_path = os.path.join(workspace, 'audio_4class_iemocap.h5')

    # split train/val/test
    sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']
    # Randomly choose 3 items for train_list
    train_list = random.sample(sessions, 3)
    # Remove the chosen items from the original list
    remaining_sessions = [session for session in sessions if session not in train_list]
    # Randomly choose 1 item for val_list
    val_list = random.sample(remaining_sessions, 1)
    # Remove the chosen item from the remaining sessions
    test_list = [session for session in remaining_sessions if session not in val_list]

    logging.info("Train List:{}".format(train_list))
    logging.info("Validation List:{}".format(val_list))
    logging.info("Test List:{}".format(test_list))


    x_train = []
    y_train = []
    x_val = []
    y_val = []
    x_test = []
    y_test = []
    duration = []
    count = 0

    with open(txt_path) as file:
        for line in file:
            audio_path = line.rstrip().split()[0]
            emo = line.rstrip().split()[1]
            if emo not in emos:
                continue
            emo = emo_check(emo)
            session = audio_path.split('/')[0]
            gender = audio_path.split('/')[-1].split('_')[0][-1]
            audio_path = os.path.join(workspace, audio_path)
            (audio, fs) = read_audio_no_unify(audio_path, target_fs=config.sample_rate)
            if session in train_list:
                x_train.append(audio)
                y_train.append(emo)
                emo_div['train'][emos[emo]] += 1
            elif session in val_list:
                x_val.append(audio)
                y_val.append(emo)
                emo_div['val'][emos[emo]] += 1
            elif session in test_list:
                x_test.append(audio)
                y_test.append(emo)
                emo_div['test'][emos[emo]] += 1

            emo_gender[gender][emos[emo]] += 1
            duration.append(librosa.get_duration(filename=audio_path))
            count += 1

    logging.info('In total of {}: {}/{}/{} samples in train/val/test'.format(count, len(y_train), len(y_val), len(y_test)))
    logging.info('the duration statis of train+val+test')
    logging.info('the mean: {} and the median: {}'.format(mean(duration), median(duration)))
    logging.info('the percentile: {}'.format(np.percentile(duration, range(0, 105, 10))))
    logging.info('emotion distribution: {}'.format(emo_div))
    logging.info('gender emotion distribution: {}'.format(emo_gender))

    x_train = np.array(x_train, dtype=object)
    x_val = np.array(x_val, dtype=object)
    x_test = np.array(x_test, dtype=object)

    y_train = np.array(y_train, dtype='int32')
    y_val = np.array(y_val, dtype='int32')
    y_test = np.array(y_test, dtype='int32')

    logging.info('finished reading audios and generating arrays')

    dt = h5py.special_dtype(vlen=np.dtype('float32'))

    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset("train_audio",  data=x_train, dtype=dt)
        hf.create_dataset("train_y", data=y_train, dtype='int32')
        hf.create_dataset("dev_audio",  data=x_val, dtype=dt)
        hf.create_dataset("dev_y", data=y_val, dtype='int32')
        hf.create_dataset("test_audio",  data=x_test, dtype=dt)
        hf.create_dataset("test_y", data=y_test, dtype='int32')

    hf.close()
    logging.info('Save to audio arrays and labels hdf5 located at {}'.format(hdf5_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=str, default='/vol/bitbucket/yc7020/fdml/workspace/iemocap')
    args = parser.parse_args()

    # Create log
    custom = '4_classes_hdf5'
    create_logging(args.workspace, custom, filemode='w')
    logging.info(args)

    iemocap_gen(args)

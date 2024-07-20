import numpy as np
import soundfile
import librosa
import os
import pandas as pd
from sklearn import metrics
from collections import Counter
import logging
import torch
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import matplotlib.ticker as ticker
import random
import config
import csv
import keras
import keras.backend as K
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score, accuracy_score


def create_folder(fd):
    if not os.path.exists(fd):
        os.makedirs(fd)


def get_filename(path):
    path = os.path.realpath(path)
    name_ext = path.split('/')[-1]
    name = os.path.splitext(name_ext)[0]
    return name


def move_data_to_gpu(x, cuda):
    if x.is_cuda:
        return x

    if 'float' in str(x.dtype):
        x = torch.Tensor(x)

    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)

    else:
        raise Exception("Error!")

    if cuda:
        x = x.cuda()

    x = Variable(x)

    return x


def create_logging(log_dir, custom, filemode):

    create_folder(log_dir)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, custom+'_'+'%04d.log' % i1)):
        i1 += 1

    log_path = os.path.join(log_dir, custom+'_'+'%04d.log' % i1)
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
        datefmt='%a, %d %b %Y %H:%M:%S',
        filename=log_path,
        filemode=filemode)

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging

def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)

    # Double the audio if the feature size is less than 256
    #if len(audio) / config.overlap <= 256:
    #    audio = np.append(audio, audio)

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs

    return audio, fs


def calculate_confusion_matrix(target, predict, classes_num):
    """Calculate confusion matrix.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
      classes_num: int, number of classes
    Outputs:
      confusion_matrix: (classes_num, classes_num)
    """

    confusion_matrix = np.zeros((classes_num, classes_num), dtype=int)
    samples_num = len(target)

    for n in range(samples_num):
        confusion_matrix[target[n], predict[n]] += 1

    return confusion_matrix


def print_confusion_matrix(confusion_matrix, labels):
    logging.info('Confusion matrix:')
    logging.info('{}'.format('\t'.join(labels)))
    for i in range(0, len(labels)):
        logging.info('{}'.format('\t'.join(map(str, confusion_matrix[i]))))


def plot_confusion_matrix(confusion_matrix, title, labels, values, path):
    """Plot confusion matrix.
    Inputs:
      confusion_matrix: matrix, (classes_num, classes_num)
      labels: list of labels
      values: list of values to be shown in diagonal
    Ouputs:
      None
    """

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)

    cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

    if labels:
        ax.set_xticklabels([''] + labels, rotation=90, ha='left')
        ax.set_yticklabels([''] + labels)
        ax.xaxis.set_ticks_position('bottom')

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    for n in range(len(values)):
        plt.text(n - 0.4, n, '{:.2f}'.format(values[n]), color='yellow')

    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.tight_layout()
    fig.savefig(path, bbox_inches='tight')
#    plt.show()


def calculate_accuracy(target, predict, classes_num, average=None):
    """Calculate accuracy.
    Inputs:
      target: integer array, (audios_num,)
      predict: integer array, (audios_num,)
    Outputs:
      accuracy: float
    """

    samples_num = len(target)

    correctness = np.zeros(classes_num)
    total = np.zeros(classes_num)
    # for accuracy
    acc_count = 0

    for n in range(samples_num):

        total[target[n]] += 1

        if target[n] == predict[n]:
            correctness[target[n]] += 1

            # for accuracy
            acc_count += 1

    accuracy = correctness / total

    if average is None:
        # return accuracy
        return acc_count / samples_num, np.mean(accuracy)
    elif average == 'macro':
        return np.mean(accuracy)

    else:
        raise Exception('Incorrect average!')


def print_accuracy(class_wise_accuracy, labels):

#    print('{:<30}{}'.format('Scene label', 'accuracy'))
#    print('------------------------------------------------')
#    for (n, label) in enumerate(labels):
#        print('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
#    print('------------------------------------------------')
#    print('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))
    logging.info('{:<30}{}'.format('Emotion label', 'accuracy'))
    logging.info('------------------------------------------------')
    for (n, label) in enumerate(labels):
        logging.info('{:<30}{:.3f}'.format(label, class_wise_accuracy[n]))
    logging.info('------------------------------------------------')
    logging.info('{:<30}{:.3f}'.format('Average', np.mean(class_wise_accuracy)))


def audio_unify(audio_wav, seq_len=int(config.audio_samples)):
    if len(audio_wav) < seq_len:
        stack_n1 = math.floor(seq_len / len(audio_wav))
        #stack_n2 = int(seq_len % len(audio_wav))
        audio_new = np.tile(audio_wav, stack_n1+1)
        audio_new = audio_new[:seq_len]

        #audio_temp = audio_wav
        #for i in range(1, stack_n1):
        #   audio_temp = np.hstack((audio_temp, audio_wav))
        #audio_new = np.hstack((audio_temp, audio_wav[0:stack_n2]))
    else:
        audio_new = audio_wav[:seq_len]

    #if True in np.isnan(audio_new):
    #    print('There is nan in the new audio')
    #if len(audio_new) != seq_len:
    #    print('Wrongly unifiying audio length!')

    return audio_new


def write_pre(pre, path):
    # pd.DataFrame(pre).to_csv(path, header=None, index=None)
    with open(path, "w") as f:
        writer = csv.writer(f)
        writer.writerows(pre)

def scoring(truth, pred):
    truth = truth.argmax(axis=1)
    pred = pred.argmax(axis=1)
    uar = recall_score(truth, pred, average='macro')
    acc = accuracy_score(truth, pred)
    confusion_mat = confusion_matrix(truth, pred, labels=list(range(7)))
    return confusion_mat, uar, acc

def metrics_uar(truth, pred):
    # truth = truth.argmax(axis=1)
    # pred = pred.argmax(axis=1)
    uar = recall_score(truth, pred, average='macro')
    # acc = accuracy_score(truth, pred)
    # confusion_mat = confusion_matrix(truth, pred, labels=list(range(7)))
    return uar


def uar(y_true, y_pred):
    """Calculate accuracy for keras model compile() for demos
    """
    y_pred = K.argmax(y_pred, axis=-1)
    y_true = K.argmax(y_true, axis=-1)
    samples_num = y_true.shape[0]
    correctness = K.zeros(7)
    total = K.zeros(7)

    for n in K.arange(samples_num):
        # print(y_true)
        # print(n)
        # print(y_true[n])
        total[y_true[n]].assign(K.get_value(total[y_true[n]]) + 1)

        if y_true[n] == y_pred[n]:
            correctness[y_true[n]].assign(K.get_value(correctness[y_true[n]]) + 1)

    for n in K.arange(7):
        if total[n].numpy() == 0.0:
            total[n].assign(1)
    accuracy = correctness / total
    # print(K.mean(accuracy))

    return K.mean(accuracy).numpy()

def split_actor(actor_lists):
    # Split the data into 0.4/0.3/0.3 for train/dev/test
    # Actor independtly
    male = []
    female = []

    # Consider the gender balance in the train/val/test
    for actor in actor_lists:
        if not actor % 2:
            male.append(actor)
        else:
            female.append(actor)
    print('male list:')
    print(male)
    print('female list:')
    print(female)

    y_male = [1] * len(male)
    y_female = [1] * len(female)

    # Split the train/val/test as 0.4/0.3/0.3
    X_male_train, X_male_test, y_male_train, y_male_test = train_test_split(male, y_male, test_size=0.25, random_state=12345)
    X_male_train, X_male_val, y_male_train, y_male_val = train_test_split(X_male_train, y_male_train, test_size=0.43, random_state=12345)

    X_female_train, X_female_test, y_female_train, y_female_test = train_test_split(female, y_female, test_size=0.25, random_state=12345)
    X_female_train, X_female_val, y_female_train, y_female_val = train_test_split(X_female_train, y_female_train, test_size=0.43, random_state=12345)

    X_train = X_male_train + X_female_train
    X_val = X_male_val + X_female_val
    X_test = X_male_test + X_female_test

    return X_train, X_val, X_test




################ Sparse Attack Unique ###########
import sys
import torch.nn.functional as F
# generator
sys.path.insert(1, os.path.join(sys.path[0], './audio_models'))
sys.path.insert(1, os.path.join(sys.path[0], '../'))
# audio pre-processing
from transformers import Wav2Vec2FeatureExtractor, AutoConfig, PretrainedConfig
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
# hugging models
from transformers import AutoModelForAudioClassification, HubertForSequenceClassification, WavLMForSequenceClassification
# end2you audio models
from audio_rnn_model import AudioRNNModel

batch_size = config.batch_size
class_num = config.num_classes
audio_len = config.audio_samples
models_menu = config.models_menu
# IEMOCAP
iemocap_class_num = config.iemocap_num_classes
iemocap_audio_len = config.iemocap_audio_samples
iemocap_models_menu = config.iemocap_models_menu

# Loss
def CWLoss(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(7).type(torch.cuda.FloatTensor)[target.long()].cuda())

    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real-other, kappa))

def CWLoss_iemocap(logits, target, kappa=-0., tar=False):
    target = torch.ones(logits.size(0)).type(torch.cuda.FloatTensor).mul(target.float())
    target_one_hot = Variable(torch.eye(4).type(torch.cuda.FloatTensor)[target.long()].cuda())

    real = torch.sum(target_one_hot*logits, 1)
    other = torch.max((1-target_one_hot)*logits - (target_one_hot*10000), 1)[0]
    kappa = torch.zeros_like(other).fill_(kappa)

    if tar:
        return torch.sum(torch.max(other-real, kappa))
    else :
        return torch.sum(torch.max(real-other, kappa))

def CrossEntropyLoss(logits, target, tar=False):
    if tar:
        # For targeted attack, use negative log-likelihood of the true class probability distribution
        loss = -F.nll_loss(F.log_softmax(logits, dim=1), target)
    else:
        # For untargeted attack, use negative log-likelihood of the predicted class probability distribution
        loss = F.nll_loss(F.log_softmax(logits, dim=1), torch.argmax(logits, dim=1))
    return loss


def get_model(name, val, workspace):
    suffix = 'val' if val else 'test'
    if name == 'wav2vec2':
        this_dir_1 = os.path.join('models_w2v_finetune', 'freeze')
        this_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=class_num)
    elif name == 'hubert':
        this_dir_1 = os.path.join('models_hu_finetune', 'freeze')
        hubert_config = AutoConfig.from_pretrained("facebook/hubert-base-ls960", num_labels=class_num)
        this_model = HubertForSequenceClassification.from_pretrained(
                "facebook/hubert-base-ls960",
                config=hubert_config,
                ignore_mismatched_sizes=True,
        )
    elif name == 'wavlm':
        this_dir_1 = os.path.join('models_wl_finetune', 'freeze')
        wavlm_config = PretrainedConfig.from_pretrained("microsoft/wavlm-base", num_labels=class_num)
        this_model = WavLMForSequenceClassification.from_pretrained(
                "microsoft/wavlm-base",
                config=wavlm_config,
                ignore_mismatched_sizes=True,
        )
    elif name == 'zhao19':
        this_dir_1 = os.path.join('models_zhao19_lstm')
        this_model = AudioRNNModel
    elif name == 'emo18':
        this_dir_1 = os.path.join('models_emo18_lstm')
        this_model = AudioRNNModel

    this_dir_2 = 'train_devel' if val else 'traindevel_test'
    this_dir_3 = models_menu[name+'_'+suffix]
    model_path = os.path.join(workspace, 'sparse_attack', this_dir_1, this_dir_2, this_dir_3)

    return this_model, model_path


def get_model_iemocap(name, val, workspace):
    suffix = 'val' if val else 'test'
    if name == 'wav2vec2':
        this_dir_1 = os.path.join('models_wav2vec2_finetune', 'freeze')
        this_model = AutoModelForAudioClassification.from_pretrained("facebook/wav2vec2-base-960h", num_labels=iemocap_class_num)
    elif name == 'wavlm':
        this_dir_1 = os.path.join('models_wavlm_finetune', 'freeze')
        wavlm_config = PretrainedConfig.from_pretrained("microsoft/wavlm-base", num_labels=iemocap_class_num)
        this_model = WavLMForSequenceClassification.from_pretrained(
                "microsoft/wavlm-base",
                config=wavlm_config,
                ignore_mismatched_sizes=True,
        )
    elif name == 'zhao19':
        this_dir_1 = os.path.join('models_zhao19')
        this_model = AudioRNNModel
    elif name == 'emo18':
        this_dir_1 = os.path.join('models_emo18')
        this_model = AudioRNNModel

    this_dir_2 = 'train_devel' if val else 'traindevel_test'
    this_dir_3 = iemocap_models_menu[name+'_'+suffix]
    model_path = os.path.join(workspace, 'sparse_attack', 'trained_models', this_dir_1, this_dir_2, this_dir_3)

    return this_model, model_path


def preprocess_function_single(examples, cuda):
    processed_examples = feature_extractor(examples, sampling_rate=config.sample_rate, return_tensors='pt')
    processed_examples = processed_examples['input_values']
    processed_examples = move_data_to_gpu(processed_examples, cuda)[0] # without the [0], it would be torch.Size([1, 16, 1, 95988])
    return processed_examples

def normalize_function(data):
    # the input data herein is the torch tensor
    means = data.mean(dim=1, keepdim=True)
    stds = data.std(dim=1, keepdim=True)
    normalized_data = (data - means) / stds
    return normalized_data



######## Wavuent related #######
def worker_init_fn(worker_id): # This is apparently needed to ensure workers have different random seeds and draw different examples!
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def get_lr(optim):
    return optim.param_groups[0]["lr"]

def set_lr(optim, lr):
    for g in optim.param_groups:
        g['lr'] = lr

def set_cyclic_lr(optimizer, it, epoch_it, cycles, min_lr, max_lr):
    cycle_length = epoch_it // cycles
    curr_cycle = min(it // cycle_length, cycles-1)
    curr_it = it - cycle_length * curr_cycle

    new_lr = min_lr + 0.5*(max_lr - min_lr)*(1 + np.cos((float(curr_it) / float(cycle_length)) * np.pi))
    set_lr(optimizer, new_lr)

import pickle
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import requests
import torch
from tqdm import tqdm
from utilities import move_data_to_gpu
import logging


def perturb_image(xs, img):
    # This function accepts one single image but one/multiple perturbation vector(s)

    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
         xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed images
    # tile = [len(xs)] + [1] * (xs.ndim + 1)
    # if len(xs) == 1:
    #     tile = [1] + [1] * 1
    # else:
    #     tile = [len(xs)] + [1] * 2
    tile = [len(xs)] + [1]
    imgs = np.tile(img.detach().cpu().numpy(), tile)
    # Make sure to floor the members of xs as int types, but not the epsilon
    # xs[:, 0] = xs[:, 0].astype(int)

    for x, img in zip(xs, imgs):
        pixels = np.split(x, len(x) // 2)
        for pixel in pixels:
            x_pos, epsilon = pixel
            img[int(x_pos)] = img[int(x_pos)] * (1 + epsilon)

    # convert the imgs back to tensor
    # imgs = move_data_to_gpu(imgs, cuda=True)
    return imgs



def evaluate_models(models, x_test, y_test):
    # x_test and y_test are dataloder
    correct_imgs = []
    network_stats = []

    for model in models:
        model_name = model.__class__.__name__
        model_params = sum(p.numel() for p in model.parameters())

        predictions = model(x_test)[0]
        predictions = predictions.detach().cpu().numpy()
        y_test = y_test.data.cpu().numpy()

        print(predictions.shape)
        print(y_test.shape)
        correct = [[model_name, i, label, np.max(pred), pred]
                   for i, (label, pred)
                   in enumerate(zip(y_test, predictions))
                   if label == np.argmax(pred)]
        accuracy = len(correct) / len(x_test)
        correct_imgs += correct
        network_stats += [[model_name, accuracy, model_params]]
    return network_stats, correct_imgs

import numpy as np
import h5py
import time
import statistics
import logging

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from utilities import move_data_to_gpu, audio_unify

# from differential_evolution import differential_evolution
from scipy.optimize import differential_evolution
import helper

class PixelAttacker:
    def __init__(self, model, dimensions, pixel_counts, eps, targeted_attack, model_type):
        # YC: adjust the dimensions here for the audio
        # Load data and model
        self.model = model
        self.dimensions = dimensions
        self.pixel_counts = pixel_counts
        self.eps = eps
        self.targeted_attack = targeted_attack
        self.model_type = model_type
        self.model.eval()

    def predict_classes(self, xs, img, target_class, model):
        ####### This function just focuses one single img and one single perturbation #########

        # Perturb the image with the given pixel(s) x and get the prediction of the model
        # Initialize with 400 perturbations! xs.shape = [400, 2]
        # logging.info("fourth checpoint is {}".format(xs.shape))
        imgs_perturbed = helper.perturb_image(xs, img)
        # logging.info('five checkpoint is {}'.format(imgs_perturbed.shape))
        imgs_perturbed = torch.from_numpy(imgs_perturbed).cuda()

        # target_class = target_class.detach().cpu().numpy()
        if self.model_type:
            # print(imgs_perturbed.shape)
            predictions = model(imgs_perturbed)[0].detach().cpu().numpy()
        else:
            predictions = model(imgs_perturbed).detach().cpu().numpy()
        # logging.info("six checkpoint is {}".format(predictions.shape))
        predictions = predictions[:, target_class]
        # logging.info("seven checkpoint is {}".format(predictions.shape))
        # This function should always be minimized, so return its complement if needed
        if self.targeted_attack:
            return 1 - predictions
        else:
            return predictions
        # return predictions if minimize else 1 - predictions

    def attack_success(self, x, img, target_class, model, verbose=False):
        ####### This function just focuses one single img and one single perturbation #########

        # Perturb the image with the given pixel(s) and get the prediction of the model
        attack_image = helper.perturb_image(x, img)
        # convert the imgs back to tensor
        attack_image = torch.from_numpy(attack_image)
        attack_image = move_data_to_gpu(attack_image, cuda=True)
        if self.model_type:
            confidence = model(attack_image)[0].detach().cpu().numpy()
        else:
            confidence = model(attack_image).detach().cpu().numpy()
        predicted_class = np.argmax(confidence)

        # If the prediction is what we want (misclassification or
        # targeted classification), return True
        if verbose:
            logging.info('Confidence:', confidence[target_class])
        if ((self.targeted_attack and predicted_class == target_class) or
                (not self.targeted_attack and predicted_class != target_class)):
            return True

    def attack(self, batch_x, target_class, maxiter=20, popsize=300, verbose=False, plot=False):
        dim_x = self.dimensions
        bounds = [(0, dim_x), (-self.eps, self.eps)] * self.pixel_counts

        # Population multiplier, in terms of the size of the perturbation vector x
        # More pixels, less need for popsize
        popmul = max(1, popsize // len(bounds))

        # Format the predict/callback functions for the differential evolution algorithm
        def predict_fn(xs):
            return self.predict_classes(xs, batch_x, target_class, self.model)

        def callback_fn(x, convergence):
            return self.attack_success(x, batch_x, target_class, self.model, verbose)

        # Call Scipy's Implementation of Differential Evolution
        attack_result = differential_evolution(predict_fn, bounds, maxiter=maxiter,
                popsize=popmul, recombination=1, atol=-1, callback=callback_fn, polish=False)

        # Calculate some useful statistics to return from this function
        # attack_result.x is the solution array

        attack_image = helper.perturb_image(attack_result.x, batch_x)
        return attack_image

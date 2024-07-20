import torch
import torch.nn as nn
import numpy as np
from rnn import RNN

from emo18 import Emo18
from zhao19 import Zhao19


class AudioModel(nn.Module):

    def __init__(self,
                 model_name:str,
                 pretrained:bool = False,
                 *args, **kwargs):
        """ Audio network model.

        Args:
            model_name (str): Name of audio model to use.
            pretrain (bool): Whether to use pretrain model (default `False`).
        """

        super(AudioModel, self).__init__()

        self.model = self._get_model(model_name)
        self.model = self.model(*args, **kwargs)
        self.num_features = self.model.num_features

    def _get_model(self, model_name):
        """ Factory method to choose audio model."""

        return {
            'emo18': Emo18,
            'zhao19': Zhao19
        }[model_name]

    def forward(self, x):
        """ Forward pass

        Args:
            x (BS x S x 1 x T)
        """
        return self.model(x)



class AudioRNNModel(nn.Module):

    def __init__(self,
                 input_size:int,
                 num_outs:int,
                 pretrained:bool = False,
                 model_name:str = None,
                 rnn_name:str=None):
        """ Convolutional recurrent neural network model.

        Args:
            input_size (int): Input size to the model.
            num_outs (int): Number of output values of the model.
            pretrained (bool): Use pretrain model (default `False`).
            model_name (str): Name of model to build (default `None`).
        """

        super(AudioRNNModel, self).__init__()
        audio_network = AudioModel(model_name=model_name, input_size=input_size)
        self.audio_model = audio_network.model
        num_out_features = audio_network.num_features
        # self.input_rnn_size = num_out_features
        self.rnn, num_out_features = self._get_rnn_model(num_out_features, rnn_name)
        self.linear = nn.Linear(num_out_features, num_outs)
        self.num_outs = num_outs
        self.rnn_name = rnn_name

    def _get_rnn_model(self, input_size:int, rnn_name:str):
        """ Builder method to get RNN instace."""

        rnn_args = {
            'input_size': input_size,
            'hidden_size': 256,
            'num_layers': 2,
            'batch_first':True,
            'bidirectional':False
        }
        if rnn_args['bidirectional']:
            return RNN(rnn_args, rnn_name), rnn_args['hidden_size'] * 2
        else:
            return RNN(rnn_args, rnn_name), rnn_args['hidden_size']

    def forward(self, x:torch.Tensor):
        """
        Args:
            x ((torch.Tensor) - BS x S x 1 x T)
        """
        if len(list(x.shape)) != 3:
            batch_size, seq_length = x.shape
            # x = x.view(batch_size*seq_length, 1)
            x = x.view(batch_size, 1, seq_length)
        audio_out = self.audio_model(x)
        audio_out = audio_out.transpose(1, 2)
        rnn_out, _ = self.rnn(audio_out)
        output = self.linear(rnn_out)
        # sample-wise
        output = torch.mean(output, dim=1)
        return output

import logging

import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F

from models.utils import Conv1DSequence
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from typing import List, Dict
import torch
import torch.nn as nn


class CNN1D_LSTM(TorchModelV2):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.original_space
        td, ti, pv = obs_space_.spaces['trade_data'], obs_space_.spaces['tech_indicators'], \
                     obs_space_.spaces['private_vars']

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        conv_filters = model_config.get('conv_filters')
        self.lstm_state_size = model_config.get("lstm_cell_size", 128)

        self.split_channel = model_config['custom_model_config'].get('split_channel', True)
        if self.split_channel:
            c, l = td.shape
            self.c, self.l = c, l
            shape = (c, l)
        else:
            self.fc1 = nn.Linear(int(np.product(td.shape)), 128)
            shape = (1, 128)

        if conv_filters is None:
            conv_filters = [80, 48]

        conv_seqs = []

        for out_channels in conv_filters:
            conv_seq = Conv1DSequence(shape[1], shape[0], out_channels, k_size=1, use_max_pool=True)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        prev_layer_size = int(np.product(shape))
        self.last_cv = conv_seqs[-1].conv
        self.conv_seqs = nn.ModuleList(conv_seqs)

        self.fc2 = nn.Linear(prev_layer_size, 32)
        prev_layer_size = 32 + ti.shape[1] + pv.shape[1]
        layers = []
        for size in hiddens[:-1]:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        self.layers = layers
        self._hidden_layers = nn.Sequential(*layers)
        self.final_layer = nn.Linear(prev_layer_size, hiddens[-1])
        self.num_outputs = hiddens[-1]
        # Holds the current "base" output (before logits layer).
        self.cuda = torch.cuda.is_available()
        self._last_batch_size = None

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict['obs']
        td, ti, pv = obs['trade_data'], obs['tech_indicators'], obs['private_vars']
        b = td.shape[0]

        if not self.split_channel:
            x = td.view(b, -1)
            x = self.fc1(x)
            x = F.relu(x)
            x = torch.unsqueeze(x, 1)
        else:
            x = td.view(b, self.c, self.l)

        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
            x = F.relu(x)

        x_flat = torch.flatten(x, start_dim=1)
        x = F.relu(x_flat)

        x = self.fc2(x)
        x = F.relu(x)

        x = torch.cat([x, ti[:, -1, :], pv[:, -1, :]], dim=1)
        x = self._hidden_layers(x)
        x = self.final_layer(x)
        x = F.relu(x)
        self._last_batch_size = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


ModelCatalog.register_custom_model("cnn1d", CNN1D_LSTM)

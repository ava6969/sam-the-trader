import logging
import os

import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from ptsdae.sdae import StackedDenoisingAutoEncoder
from models.utils import Conv1DSequence
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from typing import List, Dict
import torch
import torch.nn as nn


class SDAE_PRE(TorchModelV2):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.original_space
        td, ti, pv = obs_space_.spaces['trade_data'], obs_space_.spaces['tech_indicators'], \
                     obs_space_.spaces['private_vars']

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        sdae_in = td.shape[0] + ti.shape[0]
        dim_list = model_config['custom_model_config'].get('hiddens', [10, 16])
        dim = [sdae_in] + dim_list
        self.extract_sdae = StackedDenoisingAutoEncoder(dim)
        stringify_dim = '_'.join(map(str, dim))
        save_path = f'sdae_t_{stringify_dim}'
        save_path = os.path.join('/home/dewe/sam/sdae_trainable_model', save_path)
        assert os.path.exists(save_path), \
            f'model with config {dim} does not exist'
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extract_sdae.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        prev_layer_size = dim[-1] + pv.shape[0]
        self.fc1 = nn.Linear(prev_layer_size, prev_layer_size)
        layers = []
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size
        self.layers = layers
        self._hidden_layers = nn.Sequential(*layers)

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

        x_in = torch.cat([td, ti], dim=1)
        with torch.no_grad():
            x = self.extract_sdae.encoder(x_in).detach()
            x = x.to(self.device)

        x = torch.cat([x, pv], dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self._hidden_layers(x)
        x = F.relu(x)
        self._last_batch_size = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))


ModelCatalog.register_custom_model("sdae_pre", SDAE_PRE)

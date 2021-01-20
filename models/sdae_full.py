import glob
import logging
import os

import gym
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from torch.optim import Adam

from ptsdae.dae import DenoisingAutoencoder
from ptsdae.model import load, GaussLayer
from ptsdae.sdae import StackedDenoisingAutoEncoder
from models.utils import Conv1DSequence
from ray.rllib.models import ModelCatalog
from ptsdae.model import pretrain
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import TensorType, ModelConfigDict
from typing import List, Dict
import torch
import torch.nn as nn


class SDAE_FULL(TorchModelV2):
    def __init__(self, obs_space: gym.spaces.Space, action_space: gym.spaces.Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.shape
        self._train = model_config['custom_model_config']['train']
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        sdae_in = int(np.product(obs_space_))
        dim_list = model_config['custom_model_config'].get('hiddens', [10, 16])
        self.batch_norm = model_config['custom_model_config'].get('batch_norm', False)
        dim = [sdae_in] + dim_list
        self.extract_sdae = StackedDenoisingAutoEncoder(dim)
        self.lowest_loss = np.inf
        stringify_dim = '_'.join(map(str, dim))
        save_path = f'sdae_t_{stringify_dim}'
        self.save_path = os.path.join('/home/dewe/sam/sdae_trainable_model', save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sdae_loss = 0

        prev_layer_size = dim[-1]
        layers = []
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            if self.batch_norm:
                layers.append(nn.LayerNorm(size))
            prev_layer_size = size
        self.layers = layers
        self._hidden_layers = nn.Sequential(*layers)

        self.num_outputs = hiddens[-1]
        # Holds the current "base" output (before logits layer).
        self.cuda = torch.cuda.is_available()
        self._last_batch_size = None

    def load_model(self):
        files = glob.glob(self.save_path + '_*')
        files = sorted(files, key=lambda x: int(x[-1]))
        number_of_subautoencoders = len(self.extract_sdae.dimensions) - 1
        for index in range(number_of_subautoencoders):
            encoder, decoder = self.extract_sdae.get_stack(index)
            embedding_dimension = self.extract_sdae.dimensions[index]
            hidden_dimension = self.extract_sdae.dimensions[index + 1]
            sub_autoencoder = DenoisingAutoencoder(
                embedding_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                activation=torch.nn.ReLU()
                if index != (number_of_subautoencoders - 1)
                else None,
                corruption=GaussLayer() if index == 0 else None,
            )
            sub_autoencoder.load_state_dict(torch.load(files[index]))
            sub_autoencoder.copy_weights(encoder, decoder)

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):
        obs = input_dict['obs_flat']
        with torch.no_grad():
            x = self.extract_sdae.encoder(obs).detach()
            x = x.to(self.device)

        x = self._hidden_layers(x)
        x = F.relu(x)
        self._last_batch_size = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch_size,)))

    def custom_loss(self, policy_loss: TensorType,
                    loss_inputs: Dict[str, TensorType]) -> TensorType:
        obs = loss_inputs['obs']

        def epoch_callback(epoch, autoencoder, model_idx, loss_value, validation_loss_value):
            self.sdae_loss = loss_value
            if loss_value < self.lowest_loss:
                self.lowest_loss = loss_value
                torch.save(autoencoder.state_dict(), self.save_path + f'_{model_idx}')
                print('saved model at epoch with loss', self.lowest_loss)

        pretrain(obs,
                 self.extract_sdae,
                 1,
                 obs.shape[0],
                 silent=True,
                 optimizer=lambda model: Adam(model.parameters()),
                 epoch_callback=epoch_callback)

        return policy_loss


ModelCatalog.register_custom_model("sdae_full", SDAE_FULL)

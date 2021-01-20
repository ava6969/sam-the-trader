import logging

from ray.rllib.models.modelv2 import ModelV2
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from models.utils import ResNet, Conv2DSequence, Conv1DSequence
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from typing import List, Dict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CNN1D2D(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.original_space
        td, ti, pv = obs_space_.spaces['trade_data'], obs_space_.spaces['tech_indicators'], \
                     obs_space_.spaces['private_vars']

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])
        conv_filters = [32, 64, 64 ]
        # max_pool = [True, True, True]

        w, h = ti.shape
        shape = (1, w, h)
        self.img_shape = shape
        conv_seqs = []
        # for (out_channels, mp) in zip(conv_filters, max_pool):
        for out_channels in conv_filters:
            conv_seq = ResNet(shape, out_channels)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs.append(nn.Flatten())
        prev_layer_size_1 = int(np.product(shape))
        # conv_seqs.extend([nn.Dropout(0.25),
        #                   nn.Linear(prev_layer_size, 128),
        #                   nn.Dropout(0.5),
        #                   nn.Linear(128, 64)])
        self.conv_seqs = nn.ModuleList(conv_seqs)

        conv_seqs = []
        c, l = td.shape
        self.c, self.l = c, l
        shape = (c, l)
        conv_filters = [64, 64]
        for out_channels in conv_filters:
            conv_seq = Conv1DSequence(shape[1], shape[0], out_channels, k_size=1, use_max_pool=True)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)
        conv_seqs.append(nn.Flatten())
        prev_layer_size_2 = int(np.product(shape))
        # conv_seqs.extend([nn.Linear(prev_layer_size, 32)])
        self.conv_seqs_1 = nn.ModuleList(conv_seqs)

        prev_layer_size = prev_layer_size_1 + prev_layer_size_2 + pv.shape[1]

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
        self._last_batch = None
        self.num_outputs = prev_layer_size

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict['obs']
        td, ti, pv = obs['trade_data'], obs['tech_indicators'], obs['private_vars']
        b = td.shape[0]

        x = ti.view(b, *self.img_shape)
        y = td.view(b, self.c, self.l)

        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
            if not isinstance(conv_seq, nn.Flatten):
                x = F.relu(x)

        for conv_seq in self.conv_seqs_1:
            y = conv_seq(y)
            if not isinstance(conv_seq, nn.Flatten):
                y = F.relu(y)

        x = torch.cat([y, x, pv[:, -1, :]], dim=1)

        x = self._hidden_layers(x)
        self._last_batch = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch,)))


ModelCatalog.register_custom_model("cnn1d2d", CNN1D2D)

import logging

from ray.rllib.models.modelv2 import ModelV2
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from models.utils import Conv1DSequence
from ray.rllib.models import ModelCatalog

from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType
from typing import List, Dict
import torch
import torch.nn as nn


logger = logging.getLogger(__name__)


class Essemble_DNN(TorchRNN, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, dropout=0.2):
        TorchRNN.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        shape = int(np.product(obs_space.shape))
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [128, 32])
        conv_filters = model_config.get('conv_filters')

        self.fc1 = nn.Linear(shape, hiddens[0])

        if conv_filters is None:
            conv_filters = [80, 48]

        conv_seqs = []
        shape = (1, hiddens[0])
        for out_channels in conv_filters:
            conv_seq = Conv1DSequence(shape[1], shape[0], out_channels, k_size=1, use_max_pool=True)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        prev_layer_size = int(np.product(shape))

        self.lstm_state_size = model_config.get("lstm_cell_size", 128)
        self.l_conv = conv_seqs[-1]
        self.conv_seqs = nn.ModuleList(conv_seqs)

        self.lstm = nn.LSTM(prev_layer_size, self.lstm_state_size, num_layers=1, batch_first=True, dropout=dropout)

        self.fc2 = nn.Linear(self.lstm_state_size, hiddens[1])

        self._logits = SlimFC(
            in_size=hiddens[1],
            out_size=num_outputs,
            initializer=normc_initializer(0.01),
            activation_fn=None)

        self._value_branch = SlimFC(
            in_size=hiddens[1],
            out_size=1,
            initializer=normc_initializer(1.0),
            activation_fn=None)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(TorchRNN)
    def forward_rnn(self, inputs: TensorType, state: List[TensorType],
                    seq_lens: TensorType) -> (TensorType, List[TensorType]):

        b = inputs.shape[0]
        t = inputs.shape[1]

        in_ = inputs.view((b * t, -1))
        x = self.fc1(in_)
        x = x.view((b * t, 1, -1))
        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
            x = F.relu(x)

        x = torch.flatten(x, start_dim=1)
        x_time_ranked = torch.reshape(x, (b, t, x.shape[-1]))
        x_time_ranked, [h, c] = self.lstm(x_time_ranked, [torch.unsqueeze(state[0], 0),
                                                          torch.unsqueeze(state[1], 0)])
        x = F.relu(x_time_ranked)

        self._features = self.fc2(x)

        action_out = self._logits.forward(self._features)

        return action_out, [torch.squeeze(h, 0), torch.squeeze(c, 0)]

    # @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.

        h = [
            self.l_conv.conv.weight.new( 1, self.lstm_state_size).zero_().squeeze(0),
            self.l_conv.conv.weight.new( 1, self.lstm_state_size).zero_().squeeze(0),
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self._value_branch.forward(self._features), [-1])


ModelCatalog.register_custom_model("essemble_dnn", Essemble_DNN)

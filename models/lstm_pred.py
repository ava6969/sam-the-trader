import logging
from models.utils import LSTM
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch.nn.functional as F
from models.utils import ResNet
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.typing import TensorType
from typing import List, Dict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LSTM_NET(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.original_space
        td, ti, pv = obs_space_.spaces['trade_data'], obs_space_.spaces['tech_indicators'], \
                     obs_space_.spaces['private_vars']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm_dim = 128
        self.lstm_net = LSTM(td.shape[1], lstm_dim, 2, 1)
        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])

        prev_layer_size = lstm_dim + pv.shape[1] + ti.shape[1]
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

        self._features = None

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict['obs']
        td, ti, pv = obs['trade_data'], obs['tech_indicators'], obs['private_vars']
        b = td.shape[0]
        x = (td - torch.min(td)) / (torch.max(td) - torch.min(td))
        _, x = self.lstm_net(x)
        x.to(self.device)
        x = torch.cat([x, pv[:, -1, :], ti[:, -1, :]], dim=1)
        x = self._hidden_layers(x)
        self._last_batch = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch,)))


ModelCatalog.register_custom_model("lstm_net", LSTM_NET)

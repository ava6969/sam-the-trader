import logging

from einops import rearrange
from ray.rllib import SampleBatch
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


class ATTN_CNN(TorchModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)

        obs_space_ = obs_space.original_space
        td, ti, pv = obs_space_.spaces['trade_data'], obs_space_.spaces['tech_indicators'], \
                     obs_space_.spaces['private_vars']

        activation = model_config.get("fcnet_activation")
        hiddens = model_config.get("fcnet_hiddens", [])

        # perform conv1d conv

        conv_filters = [16, 32, 64]
        w, h = ti.shape[0] , ti.shape[1] + pv.shape[1] + td.shape[1]
        shape = (1, w, h)
        self.N = int(w*h)
        self.node_size = 64
        self.img_shape = shape
        conv_seqs = []
        for out_channels in conv_filters:
            conv_seq = Conv2DSequence(shape, out_channels, max_pool=False)
            shape = conv_seq.get_output_shape()
            conv_seqs.append(conv_seq)

        prev_layer_size = int(np.product(shape))
        self.conv_seqs = nn.ModuleList(conv_seqs)

        self.sp_coord_dim = 2
        self.n_heads = 3
        self.proj_shape = (conv_filters[-1] + self.sp_coord_dim, self.n_heads * self.node_size)
        self.k_proj = nn.Linear(*self.proj_shape)
        self.q_proj = nn.Linear(*self.proj_shape)
        self.v_proj = nn.Linear(*self.proj_shape)

        self.k_lin = nn.Linear(self.node_size, self.N)  # B
        self.q_lin = nn.Linear(self.node_size, self.N)
        self.a_lin = nn.Linear(self.N, self.N)

        self.node_shape = (self.n_heads, self.N, self.node_size)
        self.k_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.q_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)
        self.v_norm = nn.LayerNorm(self.node_shape, elementwise_affine=True)

        self.linear1 = nn.Linear(self.n_heads * self.node_size, self.node_size)
        self.norm1 = nn.LayerNorm([self.N, self.node_size], elementwise_affine=False)

        prev_layer_size = self.node_size
        layers = []
        for size in hiddens:
            layers.append(
                SlimFC(
                    in_size=prev_layer_size,
                    out_size=size,
                    initializer=normc_initializer(1.0),
                    activation_fn=activation))
            prev_layer_size = size

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.att_map = None
        self.layers = layers
        self._hidden_layers = nn.Sequential(*layers)
        self._last_batch = None
        self.num_outputs = prev_layer_size

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> (TensorType, List[TensorType]):

        obs = input_dict['obs']
        td, ti, pv = obs['trade_data'], obs['tech_indicators'], obs['private_vars']
        N = td.shape[0]

        x = torch.cat([td, ti, pv], dim=2)
        x = x.view(N, *self.img_shape)

        for conv_seq in self.conv_seqs:
            x = conv_seq(x)
            x = F.relu(x)

        _, _, cH, cW = x.shape
        xcoords = torch.arange(cW).repeat(cH, 1).float() / cW
        ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
        spatial_coords = torch.stack([xcoords, ycoords], dim=0)
        spatial_coords = spatial_coords.unsqueeze(dim=0)
        spatial_coords = spatial_coords.repeat(N, 1, 1, 1).to(self.device)
        x = torch.cat([x, spatial_coords], dim=1)
        x = x.permute(0, 2, 3, 1)
        x = x.flatten(1, 2)

        K = rearrange(self.k_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        K = self.k_norm(K)

        Q = rearrange(self.q_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        Q = self.q_norm(Q)

        V = rearrange(self.v_proj(x), "b n (head d) -> b head n d", head=self.n_heads)
        V = self.v_norm(V)
        A = torch.nn.functional.elu(self.q_lin(Q) + self.k_lin(K))  # D
        A = self.a_lin(A)
        A = torch.nn.functional.softmax(A, dim=3)
        with torch.no_grad():
            self.att_map = A.clone()  # E
        E = torch.einsum('bhfc,bhcd->bhfd', A, V)  # F
        E = rearrange(E, 'b head n d -> b n (head d)')
        E = self.linear1(E)
        E = torch.relu(E)
        E = self.norm1(E)
        E = E.max(dim=1)[0]

        x = self._hidden_layers(E)
        self._last_batch = x.shape[0]
        return x, state

    def value_function(self):
        return torch.from_numpy(np.zeros(shape=(self._last_batch,)))


ModelCatalog.register_custom_model("attn_cnn", ATTN_CNN)

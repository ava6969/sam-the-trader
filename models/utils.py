import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
import torch.nn as nn
from einops import rearrange

class SDAE(nn.Module):
    def __init__(self, dim, slope=0.0, training=True):
        super(SDAE, self).__init__()
        self.training = training
        self.in_dim = dim[0]
        self.nlayers = len(dim)-1
        self.reluslope = slope
        self.enc, self.dec = [], []
        for i in range(self.nlayers):
            self.enc.append(nn.Linear(dim[i], dim[i+1]))
            setattr(self, 'enc_{}'.format(i), self.enc[-1])
            self.dec.append(nn.Linear(dim[i+1], dim[i]))
            setattr(self, 'dec_{}'.format(i), self.dec[-1])
        self.base = []
        for i in range(self.nlayers):
            self.base.append(nn.Sequential(*self.enc[:i]))

        self.loss = nn.MSELoss()
        self.ae_optimizer = optim.Adam(self.parameters())
        self.ae_scheduler = StepLR(self.ae_optimizer, 100)
        self.loss_value = 0

    def forward(self, x):
        encoded = x

        for i, encoder in enumerate(self.enc):
            encoded = encoder(encoded)
            if i < self.nlayers-1:
                encoded = F.relu(encoded)
        decoded = encoded
        for i, decoder in reversed(list(enumerate(self.dec))):
            decoded = decoder(decoded)
            if i:
                decoded = F.relu(decoded)

        return decoded, encoded

    def full_pass(self, x):

        if not self.training:
            self.eval()
            with torch.no_grad():
                _, encoded = self.forward(x)
            return encoded
        else:
            self.train()
            decoded, encoded = self.forward(x)
            loss = self.loss(decoded, x)
            self.ae_optimizer.zero_grad()
            loss.backward()
            self.ae_optimizer.step(closure=None)
            self.ae_scheduler.step()

            self.loss_value = loss.item()

            return encoded.detach()


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)

    def forward(self, x):
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv0(x)
        x = nn.functional.relu(x)
        x = self.conv1(x)
        return x + inputs


class ResNet(nn.Module):
    def __init__(self, input_shape, out_channels, max_pool=None):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.max_pool = None if not max_pool else nn.MaxPool2d(max_pool, max_pool)
        self.res_block0 = ResidualBlock(self._out_channels)
        self.res_block1 = ResidualBlock(self._out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = self.max_pool(x)
        x = self.res_block0(x)
        x = self.res_block1(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        in_net = self.max_pool if self.max_pool else self.conv
        if self.max_pool:
            h_ = (((h + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
            w_ = (((w + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
        else:
            h_ = (((h + 2 * in_net.padding[0] - in_net.dilation[0] *
                    (in_net.kernel_size[0] - 1) - 1) / in_net.stride[0]) + 1)
            w_ = (((w + 2 * in_net.padding[1] - in_net.dilation[1] *
                    (in_net.kernel_size[1] - 1) - 1) / in_net.stride[1]) + 1)
        return self._out_channels, int(h_), int(w_)


def channel_out(in_, stride, padding, dilation, kernel_size):
    return (in_ + 2 * padding - dilation * (kernel_size - 1) - 1) / stride


class Conv1DSequence(nn.Module):
    def __init__(self, length, input_channel, out_channels, k_size=2, pad=0,
                 use_max_pool=True, max_pool_k=2, max_pool_s=2, max_pool_p=0):
        super().__init__()
        self.length = length
        self._input_channel = input_channel
        self._out_channels = out_channels
        self.conv = nn.Conv1d(in_channels=self._input_channel, out_channels=self._out_channels, kernel_size=k_size,
                              padding=pad)

        self.max_pool = nn.MaxPool1d(max_pool_k, max_pool_s, max_pool_p) if use_max_pool else None

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = self.max_pool(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        in_net = self.max_pool if self.max_pool else self.conv
        in_ =  self.length
        padding = in_net.padding
        dilation = in_net.dilation
        kernel_sz = in_net.kernel_size
        stride = in_net.stride
        Lout = ((in_ + 2 * padding - dilation * (kernel_sz - 1) - 1) // stride) + 1
        return self._out_channels, int(Lout)


class Conv2DSequence(nn.Module):
    def __init__(self, input_shape, out_channels, max_pool=None):
        super().__init__()
        self._input_shape = input_shape
        self._out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=self._input_shape[0], out_channels=self._out_channels, kernel_size=3,
                              padding=1)
        self.max_pool = None if not max_pool else nn.MaxPool2d(4, 4)

    def forward(self, x):
        x = self.conv(x)
        if self.max_pool:
            x = self.max_pool(x)
        assert x.shape[1:] == self.get_output_shape()
        return x

    def get_output_shape(self):
        _c, h, w = self._input_shape
        in_net = self.max_pool if self.max_pool else self.conv
        if self.max_pool:
            h_ = (((h + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
            w_ = (((w + 2 * in_net.padding - in_net.dilation * (in_net.kernel_size - 1) - 1) / in_net.stride) + 1)
        else:
            h_ = (((h + 2 * in_net.padding[0] - in_net.dilation[0] *
                    (in_net.kernel_size[0] - 1) - 1) / in_net.stride[0]) + 1)
            w_ = (((w + 2 * in_net.padding[1] - in_net.dilation[1] *
                    (in_net.kernel_size[1] - 1) - 1) / in_net.stride[1]) + 1)
        return self._out_channels, int(h_), int(w_)


class MultiHeadRelationalModule(torch.nn.Module):
    def __init__(self, sp_coord_dim, n_features, n_nodes, num_heads=1, node_size=64):
        super(MultiHeadRelationalModule, self).__init__()

        self.sp_coord_dim = sp_coord_dim
        self.n_heads = num_heads
        self.node_size = node_size
        self.N = n_nodes

        self.proj_shape = (n_features + self.sp_coord_dim, self.n_heads * self.node_size)
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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, x):
        if len(x.shape) == 4:
            N, _, cH, cW = x.shape
            xcoords = torch.arange(cW).repeat(cH, 1).float() / cW
            ycoords = torch.arange(cH).repeat(cW, 1).transpose(1, 0).float() / cH
            spatial_coords = torch.stack([xcoords, ycoords], dim=0)
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N, 1, 1, 1).to(self.device)
            x = torch.cat([x, spatial_coords], dim=1)
            x = x.permute(0, 2, 3, 1)
            x = x.flatten(1, 2)
        else:
            x = x.unsqueeze(1)
            N, _, c = x.shape
            spatial_coords = torch.arange(c).float() / c
            spatial_coords = spatial_coords.unsqueeze(dim=0)
            spatial_coords = spatial_coords.repeat(N, 1, 1).to(self.device)
            x = torch.cat([x, spatial_coords], dim=1)
            x = x.permute(0, 2, 1)

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
        return E


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        x.to(self.device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim,requires_grad=True).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(self.device)

        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 32, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        last_hid = out[:, -1, :]
        out = self.fc(last_hid)
        # out.size() --> 100, 10
        return out, last_hid

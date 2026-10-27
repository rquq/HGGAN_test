import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none', pad_type='reflect', sn=False, dropout=0.):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True, sn=sn)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True, sn=sn)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False, sn=sn)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = Identity()

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.dropout(dx)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False, groups=1,
                 sn=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if padding == 0:
            self.pad = Identity()
        elif pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            raise ValueError("Unsupported padding type: {}".format(pad_type))

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(4, norm_dim, 0.8)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            raise ValueError("Unsupported normalization: {}".format(norm))

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            raise ValueError("Unsupported activation: {}".format(activation))

        conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias, groups=groups)
        self.conv = nn.utils.spectral_norm(conv) if sn else conv

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DeepLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being unidirectional."""
    def __init__(
        self, input_size, hidden_size, n_layers=2,
        dropout=0., batch_first=True
    ):
        """Initialize params."""
        super(DeepLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_layer = n_layers

        self.lstm = nn.LSTM(input_size,
                            self.hidden_size,
                            n_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=self.dropout
                            )

        self.lstm.flatten_parameters()

    def forward(self, x, x_len=None):
        """Propogate input forward through the network."""
        out, _ = self.lstm(x)
        return out


class DeepBLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""
    def __init__(
        self, input_size, hidden_size, n_layers=2,
        dropout=0., batch_first=True, bidirectional=True
    ):
        """Initialize params."""
        super(DeepBLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.n_layer = n_layers
        self.bidirectional = bidirectional
        hidden_split = 2 if self.bidirectional else 1

        self.lstm = nn.LSTM(input_size,
                            self.hidden_size // hidden_split,
                            n_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=self.dropout
                            )

        self.lstm.flatten_parameters()

    def forward(self, x, x_len=None):
        """Propogate input forward through the network."""
        self.lstm.flatten_parameters()
        if x_len is None:
            x_len = torch.full((x.size(0),), x.size(1), dtype=torch.long, device='cpu')
        elif isinstance(x_len, torch.Tensor):
            x_len = torch.clamp(x_len.cpu(), min=1)
        x_pack = pack_padded_sequence(x, x_len, batch_first=self.batch_first, enforce_sorted=False)
        out_pack, _ = self.lstm(x_pack)
        out, _ = pad_packed_sequence(out_pack, batch_first=self.batch_first, total_length=x.size(1))
        return out

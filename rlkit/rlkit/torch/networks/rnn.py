import torch
from torch import nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch.core import PyTorchModule, eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer

class RNN(PyTorchModule):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            rnn_type=nn.LSTM,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.rnn = rnn_type(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=len(hidden_sizes),
            batch_first=True,
        )
        self.last_fc = nn.Linear(hidden_sizes[0], output_size)

    def forward(self, input, hx=None):
        h,hx = self.rnn(input,hx)
        output = self.last_fc(h)
        return output, hx

class ConcatRNN(RNN):
    """
    Concatenate inputs along dimension and then pass through LSTM.
    """
    def __init__(self, *args, dim=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=self.dim)
        return super().forward(flat_inputs, **kwargs)

class ConcatRNNPolicy(ConcatRNN, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, *obs, **kwargs):
        # if self.obs_normalizer:
        #     obs = self.obs_normalizer.normalize(obs)
        return super().forward(*obs, **kwargs)

    def get_action(self, *obs_np, **kwargs):
        actions = self.get_actions(*[obs[None,None,:] for obs in obs_np], **kwargs)
        return actions[0,0, :], {}

    def get_actions(self, *obs, **kwargs):
        return eval_np(self, *obs, **kwargs)[0]
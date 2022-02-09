from rlkit.torch.distributions import TanhNormal
from numpy.core.shape_base import block
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.core import torch_ify, elem_or_tuple_to_numpy
import rlkit.torch.pytorch_util as ptu
import torch
import numpy as np

class BlockPushPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            action_dim,
            *args,
            **kwargs
    ):
        super().__init__(action_dim=action_dim+1,*args, **kwargs)

    def forward(self, *inputs):
        flat_inputs = torch.cat(inputs, dim=-1)
        dist = super().forward(flat_inputs)
        mean, temp, std = dist.normal_mean[...,:3], dist.normal_mean[...,-1], dist.normal_std[...,:3]
        tool_pos = inputs[0][...,:3]
        block_pos = inputs[0][...,7:10]
        target = inputs[1]
        predicate = torch.sigmoid(-torch.exp(temp)*torch.norm(mean+block_pos - tool_pos,dim=-1))[...,None]*2
                # subtarget reached + subtarget not reached
        traj = predicate * (target-block_pos) + (1-predicate) * (mean+block_pos-tool_pos)
        return TanhNormal(traj, std)

    def get_action(self, obs):
        actions = self.get_actions(obs['raw_obs'][None], obs['goal'][None])
        return actions[0, :], {}

    def get_actions(self, *obs_np):
        dist = self(*[ptu.from_numpy(obs) for obs in obs_np])
        actions = dist.sample()
        return elem_or_tuple_to_numpy(actions)

class BlockPushOracle:
    def get_action(self, obs):
        tool_pos = obs['raw_obs'][:3]
        block_pos = obs['raw_obs'][7:10]
        target = obs['goal']
        sub_target = target + (block_pos - target)*1.25
        predicate = np.linalg.norm(sub_target - tool_pos) < .05
        traj = (target-block_pos) if predicate else (sub_target-tool_pos)
        return traj, {}

    def reset(self):
        pass
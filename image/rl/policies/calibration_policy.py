import torch as th
from .encdec_policy import EncDecPolicy
import rlkit.torch.pytorch_util as ptu
import numpy as np


class CalibrationPolicy(EncDecPolicy):
    def __init__(self, *args, env, prev_vae=None, sample=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.env = env
        self.prev_vae = prev_vae
        self.sample = sample

    def get_action(self, obs):
        with th.no_grad():
            raw_obs = obs['raw_obs']
            encoder_obs = obs.get('encoder_obs', raw_obs)
            goal_set = obs.get('goal_set')

            features = [obs['goal']]

            if self.prev_vae is not None:
                if self.incl_state:
                    features.append(encoder_obs)
                    if goal_set is not None:
                        features.append(goal_set.ravel())
                pred_features = self.prev_vae.sample(th.Tensor(np.concatenate(features)).to(ptu.device)).detach()
            else:
                pred_features = self.target

            if th.is_tensor(pred_features):
                obs['latents'] = pred_features.cpu().numpy()
            else:
                obs['latents'] = pred_features

            policy_input = [raw_obs, pred_features]
            if goal_set is not None:
                policy_input.insert(1, goal_set.ravel())

            return self.policy.get_action(*policy_input)

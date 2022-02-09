import numpy as np
import torch as th
from rlkit.torch.core import PyTorchModule
import rlkit.torch.pytorch_util as ptu
from rlkit.torch.distributions import (
    Delta
)
from rlkit.torch.networks.stochastic.distribution_generator import DistributionGenerator
import random
from sklearn.svm import LinearSVR


class EncDecPolicy(PyTorchModule):
    def __init__(self, policy, features_keys, vaes=None, incl_state=True, sample=False, latent_size=None,
                 deterministic=False, random_latent=False, window=None, prev_vae=None, prev_incl_state=False,
                 goal_baseline=False):
        super().__init__()

        self.vaes = vaes if vaes is not None else []
        self.policy = policy
        if deterministic:
            assert isinstance(policy, DistributionGenerator)
            self.policy = EncDecMakeDeterministic(self.policy)
        self.features_keys = features_keys
        self.incl_state = incl_state
        self.sample = sample
        self.latent_size = latent_size
        if self.sample:
            assert self.latent_size is not None
        self.random_latent = random_latent
        self.episode_latent = None
        self.curr_vae = None
        self.window = window if window is not None else 1
        self.past_means = []
        self.past_logvars = []

        # use encoder to map to goals for prev vae
        self.prev_vae = prev_vae
        self.prev_incl_state = prev_incl_state

        self.goal_baseline = goal_baseline
        if self.goal_baseline:
            self.x_svr_estimator = LinearSVR(max_iter=5000)
            self.y_svr_estimator = LinearSVR(max_iter=5000)

    def get_action(self, obs):
        features = [obs[k] for k in self.features_keys]
        with th.no_grad():
            raw_obs = obs['raw_obs']
            encoder_obs = obs.get('encoder_obs', raw_obs)
            goal_set = obs.get('goal_set')

            if self.random_latent:
                pred_features = self.episode_latent.detach().cpu().numpy()
            elif self.goal_baseline:
                # baseline specific to valve env
                x_pred = self.x_svr_estimator.predict(np.concatenate(features)[None])[0]
                y_pred = self.y_svr_estimator.predict(np.concatenate(features)[None])[0]
                self.past_means.append([x_pred, y_pred])
                self.past_means = self.past_means[-self.window:]
                avg_pred = np.mean(self.past_means, axis=0)
                valve_pos = encoder_obs[-3:]
                valve_xy = np.delete(valve_pos, 1)
                avg_pred = avg_pred - valve_xy
                angle_pred = np.arctan2(avg_pred[1], -avg_pred[0])
                prev_encoder_inputs = [th.Tensor([np.sin(angle_pred), np.cos(angle_pred)]).to(ptu.device)]
                if self.prev_incl_state:
                    prev_encoder_inputs.append(th.Tensor(encoder_obs).to(ptu.device))
                pred_features, _ = self.prev_vae.encode(th.cat(prev_encoder_inputs))
                pred_features = pred_features.cpu().numpy()

            elif len(self.vaes):
                if self.incl_state:
                    features.append(encoder_obs)
                    if goal_set is not None:
                        features.append(goal_set.ravel())
                encoder_input = th.Tensor(np.concatenate(features)).to(ptu.device)
                mean, logvar = self.curr_vae.encode(encoder_input)
                self.past_means.append(mean)
                self.past_logvars.append(logvar)

                self.past_means = self.past_means[-self.window:]
                self.past_logvars = self.past_logvars[-self.window:]

                # use current encoder to map to latent
                if self.prev_vae is None:
                    mean, sigma_squared = self._product_of_gaussians(self.past_means, self.past_logvars)

                    if self.sample:
                        posterior = th.distributions.Normal(mean, th.sqrt(sigma_squared))
                        pred_features = posterior.rsample()
                    else:
                        pred_features = mean

                # use current encoder to map to goal for prev vae
                else:
                    prev_encoder_inputs = []
                    prev_encoder_inputs.append(th.mean(th.stack(self.past_means), dim=0))
                    if self.prev_incl_state:
                        prev_encoder_inputs.append(th.Tensor(encoder_obs).to(ptu.device))
                    pred_features, _ = self.prev_vae.encode(th.cat(prev_encoder_inputs))

                pred_features = pred_features.cpu().numpy()

            else:
                pred_features = np.concatenate(features)

            obs['latents'] = pred_features

            policy_input = [raw_obs, pred_features]
            if goal_set is not None:
                policy_input.insert(1, goal_set.ravel())
            action = self.policy.get_action(*policy_input)
            return action

    def reset(self):
        if self.random_latent:
            self.episode_latent = th.normal(ptu.zeros(self.latent_size), 1).to(ptu.device)
        self.policy.reset()
        if len(self.vaes):
            self.curr_vae = random.choice(self.vaes)
        self.past_means = []
        self.past_logvars = []

    def _product_of_gaussians(self, means, logvars):
        sigmas_squared = th.clamp(th.exp(th.stack(logvars)), min=1e-7)
        sigma_squared = 1. / th.sum(th.reciprocal(sigmas_squared), dim=0)
        mean = sigma_squared * th.sum(th.stack(means) / sigmas_squared, dim=0)
        return mean, sigma_squared

class EncDecMakeDeterministic(PyTorchModule):
    def __init__(
            self,
            policy,
    ):
        super().__init__()
        self.policy = policy

    def forward(self, *args, **kwargs):
        dist = self.policy.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())

    def get_action(self, *obs_np):
        return self.policy.get_action(*obs_np)

    def get_actions(self, *obs_np):
        return self.policy.get_actions()

    def reset(self):
        self.policy.reset()

from collections import OrderedDict, namedtuple
from typing import Tuple
from rlkit.torch.sac.sac import SACTrainer

import numpy as np
import torch
import torch.optim as optim
from rlkit.core.loss import LossStatistics

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.core.logging import add_prefix
import gtimer as gt

SACLosses = namedtuple(
    'SACLosses',
    'policy_loss qf1_loss qf2_loss alpha_loss kl_loss',
)


class EncDecSACTrainer(SACTrainer):
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,
            latent_size,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            encoder_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            target_update_period=1,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            vae=None,
            beta=0.01,
            sample=True,
            incl_state=False
    ):
        super().__init__(env,
                         policy,
                         qf1,
                         qf2,
                         target_qf1,
                         target_qf2,
                         discount,
                         reward_scale,
                         policy_lr,
                         qf_lr,
                         optimizer_class,
                         soft_target_tau,
                         target_update_period,
                         plotter,
                         render_eval_paths,
                         use_automatic_entropy_tuning,
                         target_entropy)

        self.vae = vae
        self.beta = beta
        self.sample = sample
        self.latent_size = latent_size
        self.incl_state = incl_state

        if self.vae is not None:
            self.encoder_optimizer = optimizer_class(
                self.vae.encoder.parameters(),
                lr=encoder_lr,
            )

    def train_from_torch(self, batch):
        gt.blank_stamp()
        losses, stats = self.compute_loss(
            batch,
            skip_statistics=not self._need_to_update_eval_statistics,
        )
        """
        Update networks
        """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            losses.alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        if self.vae is not None:
            self.encoder_optimizer.zero_grad()
        (losses.policy_loss + self.beta * losses.kl_loss).backward()
        self.policy_optimizer.step()
        if self.vae is not None:
            self.encoder_optimizer.step()

        self.qf1_optimizer.zero_grad()
        losses.qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        losses.qf2_loss.backward()
        self.qf2_optimizer.step()

        self._n_train_steps_total += 1

        self.try_update_target_networks()
        if self._need_to_update_eval_statistics:
            self.eval_statistics = stats
            # Compute statistics using only one batch per epoch
            self._need_to_update_eval_statistics = False
        gt.stamp('sac training', unique=False)

    def compute_loss(
            self,
            batch,
            skip_statistics=False,
    ) -> Tuple[SACLosses, LossStatistics]:
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']
        curr_goal = batch['curr_goal']
        next_goal = batch['next_goal']
        curr_goal_set = batch.get('curr_goal_set')
        next_goal_set = batch.get('next_goal_set')

        batch_size = obs.shape[0]
        has_goal_set = curr_goal_set is not None

        eps = torch.normal(ptu.zeros((batch_size, self.latent_size)), 1) if self.sample else None
        curr_encoder_features = [curr_goal]
        next_encoder_features = [next_goal]
        if self.incl_state:
            encoder_obs = batch.get('curr_encoder_obs', obs)
            next_encoder_obs = batch.get('next_encoder_obs', next_obs)
            curr_encoder_features.append(encoder_obs)
            next_encoder_features.append(next_encoder_obs)
            if has_goal_set:
                curr_encoder_features.append(curr_goal_set.reshape((batch_size, -1)))
                next_encoder_features.append(next_goal_set.reshape((batch_size, -1)))

        if self.vae is not None:
            curr_latent, kl_loss = self.vae.sample(torch.cat(curr_encoder_features, dim=1), eps=eps, return_kl=True)
            next_latent = self.vae.sample(torch.cat(next_encoder_features, dim=1), eps=None, return_kl=False)

            next_latent = next_latent.detach()

        else:
            curr_latent, next_latent = curr_goal, next_goal
            kl_loss = ptu.zeros(1, requires_grad=False)

        """
        Policy and Alpha Loss
        """

        if has_goal_set:
            curr_goal_set_flat = curr_goal_set.reshape((batch_size, -1))
            curr_policy_features = [obs, curr_goal_set_flat, curr_latent]
        else:
            curr_policy_features = [obs, curr_latent]

        dist = self.policy(*curr_policy_features)

        new_obs_actions, log_pi = dist.rsample_and_logprob()
        log_pi = log_pi.unsqueeze(-1)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        if has_goal_set:
            new_qf_features = [obs, curr_goal_set_flat, curr_goal, new_obs_actions]
        else:
            new_qf_features = [obs, curr_goal, new_obs_actions]
        q_new_actions = torch.min(
            self.qf1(*new_qf_features),
            self.qf2(*new_qf_features),
        )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """
        QF Loss
        """

        if has_goal_set:
            curr_qf_features = [obs, curr_goal_set_flat, curr_goal, actions]
            next_goal_set_flat = next_goal_set.reshape((batch_size, -1))
            next_policy_features = [next_obs, next_goal_set_flat, next_latent]
        else:
            curr_qf_features = [obs, curr_goal, actions]
            next_policy_features = [next_obs, next_latent]

        q1_pred = self.qf1(*curr_qf_features)
        q2_pred = self.qf2(*curr_qf_features)
        next_dist = self.policy(*next_policy_features)

        new_next_actions, new_log_pi = next_dist.rsample_and_logprob()
        new_log_pi = new_log_pi.unsqueeze(-1)

        if has_goal_set:
            next_qf_features = [next_obs, next_goal_set_flat, next_goal, new_next_actions]
        else:
            next_qf_features = [next_obs, next_goal, new_next_actions]

        target_q_values = torch.min(
            self.target_qf1(*next_qf_features),
            self.target_qf2(*next_qf_features),
        ) - alpha * new_log_pi

        q_target = self.reward_scale * rewards + (1. - terminals) * self.discount * target_q_values

        qf1_loss = self.qf_criterion(q1_pred, q_target.detach())
        qf2_loss = self.qf_criterion(q2_pred, q_target.detach())

        """
        Save some statistics for eval
        """
        eval_statistics = OrderedDict()
        if not skip_statistics:
            eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
            eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            eval_statistics['KL Loss'] = np.mean(ptu.get_numpy(
                kl_loss
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q2 Predictions',
                ptu.get_numpy(q2_pred),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            policy_statistics = add_prefix(dist.get_diagnostics(), "policy/")
            eval_statistics.update(policy_statistics)
            if self.use_automatic_entropy_tuning:
                eval_statistics['Alpha'] = alpha.item()
                eval_statistics['Alpha Loss'] = alpha_loss.item()

        loss = SACLosses(
            policy_loss=policy_loss,
            qf1_loss=qf1_loss,
            qf2_loss=qf2_loss,
            alpha_loss=alpha_loss,
            kl_loss=kl_loss
        )

        return loss, eval_statistics

    def get_diagnostics(self):
        stats = super().get_diagnostics()
        stats.update(self.eval_statistics)
        return stats

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        nets = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        if self.vae is not None:
            nets.append(self.vae)
        return nets

    @property
    def optimizers(self):
        opts = [
            self.alpha_optimizer,
            self.qf1_optimizer,
            self.qf2_optimizer,
            self.policy_optimizer,
        ]
        if self.vae is not None:
            opts.append(self.encoder_optimizer)
        return opts

    def get_snapshot(self):
        snapshot = dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )
        if self.vae is not None:
            snapshot['vae'] = self.vae
        return snapshot

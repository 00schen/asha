import numpy as np
from numpy.linalg import norm
import pybullet as p

class IdentityPolicy:
    """Assuming simulated keyboard, directly return the target value"""
    def __init__(self,env):
        self.base_env = env.base_env
        self.size = env.action_space.low.size

    def get_action(self, obs):
        action = obs['target']
        if np.count_nonzero(action):
            self.action = action
        else:
            action = self.action
        action = self.trajectory(action)

        obs['latents'] = np.zeros(4)

        return action, {}

    def joint(self, action):
        clip_by_norm = lambda traj, limit: traj / max(1e-4, norm(traj)) * np.clip(norm(traj), None, limit)
        action = clip_by_norm(action, 1)
        return action

    def target(self, coor):
        base_env = self.base_env
        joint_states = p.getJointStates(base_env.robot, jointIndices=base_env.robot_left_arm_joint_indices,
                                        physicsClientId=base_env.id)
        joint_positions = np.array([x[0] for x in joint_states])

        link_pos = p.getLinkState(base_env.robot, 13, computeForwardKinematics=True, physicsClientId=base_env.id)[0]
        new_pos = np.array(coor) + np.array(link_pos) - base_env.tool_pos

        new_joint_positions = np.array(
            p.calculateInverseKinematics(base_env.robot, 13, new_pos, physicsClientId=base_env.id))
        new_joint_positions = new_joint_positions[:7]
        action = new_joint_positions - joint_positions
        return self.joint(action)

    def trajectory(self, traj):
        clip_by_norm = lambda traj, min_l=None, max_l=None: traj / max(1e-4, norm(traj)) * np.clip(norm(traj),
                                                                                                    min_l, max_l)
        traj = clip_by_norm(traj, .07, .1)
        return self.target(self.base_env.tool_pos + traj)

    def disc_traj(self, action):
        index = np.argmax(action)
        traj = [
            np.array((-1, 0, 0)),
            np.array((1, 0, 0)),
            np.array((0, -1, 0)),
            np.array((0, 1, 0)),
            np.array((0, 0, -1)),
            np.array((0, 0, 1)),
        ][index]
        return self.trajectory(traj)

    def reset(self):
        self.action = np.zeros(self.size)

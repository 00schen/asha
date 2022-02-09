# Environment is not present in original assistive_gym library at https://github.com/Healthcare-Robotics/assistive-gym

import os
from gym import spaces
import numpy as np
import pybullet as p
from itertools import product
from numpy.linalg import norm
from .env import AssistiveEnv
from gym.utils import seeding
from collections import OrderedDict

reach_arena = (np.array([-.25, -.5, 1]), np.array([.6, .4, .2]))
default_orientation = p.getQuaternionFromEuler([0, 0, 0])


class BottleEnv(AssistiveEnv):
    def __init__(self, robot_type='jaco', success_dist=.05, target_indices=None, session_goal=False, frame_skip=5,
                 capture_frames=False, stochastic=True, debug=False, step_limit=200,**kwargs):
        super(BottleEnv, self).__init__(robot_type=robot_type, task='reaching', frame_skip=frame_skip, time_step=0.02,
                                        action_robot_len=7, obs_robot_len=14)
        self.observation_space = spaces.Box(-np.inf, np.inf, (7 + 3 + 6 + 7,), dtype=np.float32)
        self.num_targets = 4
        self.success_dist = success_dist
        self.debug = debug
        self.stochastic = stochastic
        self.goal_feat = ['target_pos', ]  # Just an FYI
        self.feature_sizes = OrderedDict({'goal': 3})
        self.session_goal = session_goal
        self.target_indices = list(np.arange(self.num_targets))
        self.table_offset = None

        self.wall_color = None
        self.step_limit = step_limit
        self.curr_step = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.init_pos_random, _ = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.curr_step += 1
        old_tool_pos = self.tool_pos

        self.take_step(action, robot_arm='left', gains=self.config('robot_gains'), forces=self.config('robot_forces'))
        self.move_slide_cabinet()

        self.task_success = norm(self.tool_pos - self.target_pos) < self.success_dist
        obs = self._get_obs([0])

        if self.task_success:
            target_color = [0, 1, 0, 1]
            p.changeVisualShape(self.target, -1, rgbaColor=target_color)
        elif self.wrong_goal_reached() or self.curr_step >= self.step_limit:
            target_color = [1, 0, 0, 1]
            p.changeVisualShape(self.target, -1, rgbaColor=target_color)

        reward = self.task_success
        info = {
            'task_success': self.task_success,
            'old_tool_pos': old_tool_pos,
            'target_index': self.target_index,

            'door_open': self.door_open,
            'door_pos': self.door_pos,
            'shelf_pos': self.shelf_pos,
            'tool_pos': self.tool_pos,
            'sub_target': self.sub_target_pos.copy(),
            'target_pos': self.target_pos,
            'unique_targets': self.bottle_poses,
        }
        done = False

        return obs, reward, done, info

    def move_slide_cabinet(self):
        robot_joint_position = p.getJointStates(self.shelf, jointIndices=[0], physicsClientId=self.id)[0][0]
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=self.shelf, linkIndexB=1, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=self.shelf, linkIndexB=1, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.robot, bodyB=self.shelf, linkIndexB=0, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=self.shelf, linkIndexB=0, physicsClientId=self.id)
        if len(contacts) == 0 or norm(self.tool_pos - self.door_pos) > .1:
            return 0, 0

        normal = contacts[0][7]
        c_F = -normal[0]
        k = .002
        w = k * np.sign(c_F) * np.sqrt(abs(c_F))
        for _ in range(self.frame_skip):
            robot_joint_position += w
        robot_joint_position = np.clip(robot_joint_position, *self.slide_range)
        p.resetJointState(self.shelf, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

    def _get_obs(self, forces):
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices,
                                              physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        robot_obs = dict(
            raw_obs=np.concatenate([self.tool_pos, self.tool_orient, self.door_pos, *self.bottle_poses,
                                    robot_joint_positions]),
            hindsight_goal=np.concatenate([self.tool_pos, [0]]),
            goal=self.goal.copy(),
        )
        return robot_obs

    def reset(self):
        self.task_success = False

        """set up standard environment"""
        self.setup_timing()
        _human, self.wheelchair, self.robot, self.robot_lower_limits, self.robot_upper_limits, _human_lower_limits, _human_upper_limits, self.robot_right_arm_joint_indices, self.robot_left_arm_joint_indices, self.gender \
            = self.world_creation.create_new_world(furniture_type='wheelchair', init_human=False,
                                                   static_human_base=True, human_impairment='random',
                                                   print_joints=False, gender='random')
        self.robot_lower_limits = self.robot_lower_limits[self.robot_left_arm_joint_indices]
        self.robot_upper_limits = self.robot_upper_limits[self.robot_left_arm_joint_indices]
        self.reset_robot_joints()
        wheelchair_pos, wheelchair_orient = p.getBasePositionAndOrientation(self.wheelchair, physicsClientId=self.id)
        p.resetBasePositionAndOrientation(self.robot, np.array(wheelchair_pos) + np.array([-0.35, -0.3, 0.3]),
                                          p.getQuaternionFromEuler([0, 0, -np.pi / 2.0], physicsClientId=self.id),
                                          physicsClientId=self.id)
        base_pos, base_orient = p.getBasePositionAndOrientation(self.robot, physicsClientId=self.id)
        self.human_controllable_joint_indices = []
        self.human_lower_limits = np.array([])
        self.human_upper_limits = np.array([])

        """set up target and initial robot position"""
        if not self.session_goal:
            self.set_target_index()  # instance override in demos
            self.reset_noise()

        """set up shelf environment objects"""
        self.table_pos = table_pos = np.array([0, -1.05, 0])
        if self.stochastic:
            self.table_pos = table_pos = table_pos + self.table_noise

        self.table = p.loadURDF(os.path.join(self.world_creation.directory, 'table', 'table_tall.urdf'),
                                basePosition=table_pos, baseOrientation=default_orientation, physicsClientId=self.id)

        self.init_robot_arm()
        self.generate_target()

        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[4, .1, 1])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[4, .1, 1], rgbaColor=self.wall_color)

        wall_pos, wall_orient = np.array([0., -2., 1.]), np.array([0, 0, 0, 1])
        self.wall = p.createMultiBody(basePosition=wall_pos, baseOrientation=wall_orient,
                                      baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual,
                                      physicsClientId=self.id)

        """configure pybullet"""
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
        # Enable rendering
        p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=-30,
                                     cameraTargetPosition=[0, -.25, 1.0], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.task_success = 0
        self.goal = np.array(self.target_pos)
        self.curr_step = 0

        return self._get_obs([0])

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])

        if self.stochastic:
            self.init_pos += self.init_pos_random.uniform([-0.4, -0.1, -0.1], [0.4, 0.1, 0.1], size=3)

    def init_robot_arm(self):
        self.init_start_pos()
        init_orient = p.getQuaternionFromEuler(np.array([0, np.pi / 2.0, 0]), physicsClientId=self.id)
        self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation,
                                     self.robot_left_arm_joint_indices, self.robot_lower_limits,
                                     self.robot_upper_limits,
                                     ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100,
                                     max_ik_random_restarts=10, random_restart_threshold=0.03, step_sim=True)
        self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
        self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001] * 3, pos_offset=[0, 0, 0.02],
                                                  orient_offset=p.getQuaternionFromEuler([0, -np.pi / 2.0, 0],
                                                                                         physicsClientId=self.id),
                                                  maximal=False)

    def get_random_target(self):
        return np.concatenate(self.bottle_poses[np.random.randint(2)])

    def set_target_index(self, index=None):
        if index is None:
            self.target_index = self.np_random.choice(self.target_indices)
        else:
            self.target_index = index

        self.unique_index = self.target_index // 2

    def reset_noise(self):
        offset = self.np_random.choice((0.1, 0)) if self.table_offset is None else self.table_offset
        self.table_noise = self.np_random.uniform([-0.25, -0.05, 0], [0.15, 0.05, 0], size=3)
        self.table_noise[1] = self.table_noise[1] + offset

    def wrong_goal_reached(self):
        return norm(self.tool_pos - self.wrong_target_pos) < self.success_dist

    def calibrate_mode(self, calibrate, split):
        self.wall_color = [255 / 255, 187 / 255, 120 / 255, 1] if calibrate else None
        self.table_offset = 0.1 if split else 0

    def generate_target(self):
        self.shelf_pos = self.table_pos + np.array([0, .1, 1])
        if self.target_index % 2:
            cabinet_urdf, self.slide_range = 'slide_cabinet_left.urdf', (-.3, 0)
        else:
            cabinet_urdf, self.slide_range = 'slide_cabinet_right.urdf', (0, .3)
        self.shelf = p.loadURDF(os.path.join(self.world_creation.directory, 'slide_cabinet', cabinet_urdf),
                                basePosition=self.shelf_pos, baseOrientation=default_orientation, globalScaling=1,
                                physicsClientId=self.id, useFixedBase=True)

        bottle_poses = []
        for increment in product(np.linspace(-.15, .15, num=2), [0], [-.15]):
            bottle_pos, bottle_orient = p.multiplyTransforms(self.shelf_pos, default_orientation, increment,
                                                             default_orientation, physicsClientId=self.id)
            bottle_poses.append(bottle_pos + np.array([0, 0, .05]))
        self.bottle_poses = np.array(bottle_poses)

        target_pos = self.target_pos = bottle_poses[self.target_index // 2]
        wrong_target_pos = self.wrong_target_pos = bottle_poses[1 - (self.target_index // 2)]

        bottle_pos = target_pos + np.array([0, 0, -.05])
        self.bottle = p.loadURDF(os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
                                 basePosition=bottle_pos, useFixedBase=True,
                                 baseOrientation=default_orientation, globalScaling=.01,
                                 physicsClientId=self.id)

        wrong_bottle_pos = wrong_target_pos + np.array([0, 0, -.05])
        self.wrong_bottle = p.loadURDF(
            os.path.join(self.world_creation.directory, 'bottle', 'bottle.urdf'),
            basePosition=wrong_bottle_pos, useFixedBase=True,
            baseOrientation=default_orientation, globalScaling=.01,
            physicsClientId=self.id)

        sphere_collision = -1

        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 0, 1, 1],
                                            physicsClientId=self.id)
        self.target = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                        baseVisualShapeIndex=sphere_visual, basePosition=target_pos,
                                        useMaximalCoordinates=False, physicsClientId=self.id)
        self.slide = p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                       basePosition=self.door_pos,
                                       useMaximalCoordinates=False, physicsClientId=self.id)
        self.update_targets()

    def update_targets(self):
        p.resetBasePositionAndOrientation(self.slide, self.door_pos, [0, 0, 0, 1], physicsClientId=self.id)

    @property
    def tool_pos(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[0])

    @property
    def tool_orient(self):
        return np.array(p.getBasePositionAndOrientation(self.tool, physicsClientId=self.id)[1])

    @property
    def door_pos(self):
        location = p.getLinkState(self.shelf, 1, computeForwardKinematics=True, physicsClientId=self.id)[:2]
        door_pos = [.15, .17, 0] if self.target_index % 2 else [-.15, .17, 0]
        return np.array(p.multiplyTransforms(*location, door_pos, [0, 0, 0, 1])[0])

    @property
    def door_open(self):
        final_door_pos = (np.array([-.15, .17, 0]) if self.target_index // 2 else np.array(
            [.15, .17, 0])) + self.shelf_pos
        return norm(self.door_pos - final_door_pos) < .02

    @property
    def sub_target_pos(self):
        final_door_pos = (np.array([-.15, .17, 0]) if self.target_index // 2 else np.array(
            [.15, .17, 0])) + self.shelf_pos
        return self.target_pos if self.door_open else final_door_pos


class BottleJacoEnv(BottleEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_type='jaco', **kwargs)

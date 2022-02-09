# Environment is not present in original assistive_gym library at https://github.com/Healthcare-Robotics/assistive-gym

import os
from gym import spaces
import numpy as np
import pybullet as p
from numpy.linalg import norm
from .env import AssistiveEnv
from gym.utils import seeding

LOW_LIMIT = -1
HIGH_LIMIT = .2


class LightSwitchEnv(AssistiveEnv):
    def __init__(self, message_indices=None, success_dist=.03, session_goal=False, frame_skip=5, robot_type='jaco',
                 capture_frames=False, stochastic=True, debug=False, target_indices=None, num_targets=5,
                 step_limit=200, **kwargs):
        super(LightSwitchEnv, self).__init__(robot_type=robot_type, task='switch', frame_skip=frame_skip,
                                             time_step=0.02, action_robot_len=7, obs_robot_len=18)
        self.success_dist = success_dist
        self.num_targets = num_targets
        self.messages = np.array([' '.join((['1'] * i) + ['0'] + ['1'] * (self.num_targets - i - 1))
                                  for i in range(self.num_targets)])

        self.switch_p = 1
        self.capture_frames = capture_frames
        self.debug = debug
        self.stochastic = stochastic
        self.session_goal = session_goal
        self.wall_offset = None
        obs_size = 4 + 3 + 7
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_size,), dtype=np.float32)

        self.feature_sizes = {'goal': 3}
        if target_indices is None:
            self.target_indices = list(np.arange(self.num_targets))
        else:
            for i in target_indices:
                assert 0 <= i < self.num_targets
            self.target_indices = target_indices

        self.goal_set_shape = (self.num_targets, 4)
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
        angle_dirs = np.zeros(len(self.switches))
        reward_switch = 0
        angle_diffs = []
        self.lever_angles = []

        for i, switch in enumerate(self.switches):
            angle_dirs[i], angle_diff = self.move_lever(switch)

            ### Debugging: auto flip switch ###
            if self.debug:
                tool_pos1 = np.array(
                    p.getLinkState(self.tool, 0, computeForwardKinematics=True, physicsClientId=self.id)[0])
                if (norm(self.tool_pos - self.target_pos1[i]) < .07 or norm(tool_pos1 - self.target_pos1[i]) < .1) \
                        or (
                        norm(self.tool_pos - self.target_pos1[i]) < .07 or norm(tool_pos1 - self.target_pos1[i]) < .1):
                    # for switch1 in self.switches:
                    if self.target_string[i] == 0:
                        p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
                    else:
                        p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)
                    self.update_targets()

            lever_angle = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
            self.lever_angles.append(lever_angle)
            angle_diffs.append(angle_diff)
            if lever_angle < LOW_LIMIT + .1:
                self.current_string[i] = 0
            elif lever_angle > HIGH_LIMIT - .1:
                self.current_string[i] = 1
            else:
                self.current_string[i] = 1

            if self.target_string[i] == 0:
                reward_switch += -abs(LOW_LIMIT - lever_angle)
            else:
                reward_switch += -abs(HIGH_LIMIT - lever_angle)

            if self.target_string[i] == self.current_string[i]:
                self.update_targets()

        task_success = np.all(np.equal(self.current_string, self.target_string))
        self.task_success = task_success
        if self.task_success:
            color = [0, 1, 0, 1]
        elif self.wrong_goal_reached() or self.curr_step >= self.step_limit:
            color = [1, 0, 0, 1]
        else:
            color = [0, 0, 1, 1]
        p.changeVisualShape(self.targets1[self.target_index], -1, rgbaColor=color)

        obs = self._get_obs([0])

        _, _, _, bad_contact_count = self.get_total_force()
        target_indices = np.nonzero(np.not_equal(self.target_string, self.current_string))[0]
        if len(target_indices) > 0:
            reward_dist = -norm(self.tool_pos - self.target_pos[target_indices[0]])
        else:
            reward_dist = 0
        reward = reward_dist + reward_switch
        _, switch_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)
        info = {
            'task_success': self.task_success,
            'num_correct': np.count_nonzero(np.equal(self.target_string, self.current_string)),
            'angle_dir': angle_dirs,
            'angle_diff': angle_diffs,
            'tool_pos': self.tool_pos,
            'tool_orient': self.tool_orient,
            'old_tool_pos': old_tool_pos,
            'ineff_contact': bad_contact_count,

            'target_index': self.target_index,
            'unique_index': self.target_index,
            'lever_angle': self.lever_angles.copy(),
            'target_string': self.target_string.copy(),
            'current_string': self.current_string.copy(),
            'switch_pos': np.array(self.target_pos).copy(),
            'aux_switch_pos': np.array(self.target_pos1).copy(),
            'switch_orient': switch_orient,
        }
        if self.capture_frames:
            frame = self.get_frame()
            info['frame'] = frame
        done = False
        return obs, reward, done, info

    def move_lever(self, switch):
        switch_pos, switch_orient = p.getLinkState(switch, 0)[:2]
        old_j_pos = robot_joint_position = p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
        contacts = p.getContactPoints(bodyA=self.robot, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
        contacts += p.getContactPoints(bodyA=self.tool, bodyB=switch, linkIndexB=0, physicsClientId=self.id)
        if len(contacts) == 0:
            return 0, 0

        normal = contacts[0][7]
        joint_pos, __ = p.multiplyTransforms(switch_pos, switch_orient, p.getJointInfo(switch, 0, self.id)[14],
                                             p.getQuaternionFromEuler([0, 0, 0]), physicsClientId=self.id)
        radius = np.array(contacts[0][6]) - np.array(joint_pos)
        axis, _ = p.multiplyTransforms(np.zeros(3), switch_orient, [1, 0, 0], p.getQuaternionFromEuler([0, 0, 0]),
                                       physicsClientId=self.id)
        centripedal = np.cross(axis, radius)
        c_F = np.dot(normal, centripedal) / norm(centripedal)
        k = -.2
        w = k * np.sign(c_F) * np.sqrt(abs(c_F)) * norm(radius)

        for _ in range(self.frame_skip):
            robot_joint_position += w

        robot_joint_position = np.clip(robot_joint_position, LOW_LIMIT, HIGH_LIMIT)
        p.resetJointState(switch, jointIndex=0, targetValue=robot_joint_position, physicsClientId=self.id)

        return w, robot_joint_position - old_j_pos

    def get_total_force(self):
        tool_force = 0
        tool_force_at_target = 0
        target_contact_pos = None
        bad_contact_count = 0
        for i in range(len(self.switches)):
            if self.target_string[i] == self.current_string[i]:
                for c in p.getContactPoints(bodyA=self.tool, bodyB=self.switches[i], physicsClientId=self.id):
                    bad_contact_count += 1
        return tool_force, tool_force_at_target, target_contact_pos, bad_contact_count

    def _get_obs(self, forces):
        state = p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)
        tool_pos = np.array(state[0])
        tool_orient = np.array(state[1])  # Quaternions
        robot_joint_states = p.getJointStates(self.robot, jointIndices=self.robot_left_arm_joint_indices,
                                              physicsClientId=self.id)
        robot_joint_positions = np.array([x[0] for x in robot_joint_states])

        obs_features = [tool_orient, tool_pos, robot_joint_positions]

        robot_obs = dict(
            raw_obs=np.concatenate(obs_features),
            hindsight_goal=np.zeros(3),
            goal=self.goal.copy(),
            goal_set=np.concatenate((self.goal_positions, np.array(self.lever_angles)[:, None]),
                                    axis=1)
        )
        return robot_obs

    # return if a switch other than the target switch was flipped, assumes all switches start not flipped
    def wrong_goal_reached(self):
        return np.sum(self.current_string != self.target_string) > 1

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
        self.human_controllable_joint_indices = []
        self.human_lower_limits = np.array([])
        self.human_upper_limits = np.array([])

        """set up target and initial robot position (objects set up with target)"""
        if not self.session_goal:
            self.set_target_index()  # instance override in demos
            self.reset_noise()
        self.generate_target()
        self.init_robot_arm()

        """configure pybullet"""
        p.setGravity(0, 0, 0, physicsClientId=self.id)
        p.setPhysicsEngineParameter(numSubSteps=5, numSolverIterations=10, physicsClientId=self.id)
        # Enable rendering
        p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=180, cameraPitch=0,
                                     cameraTargetPosition=[0, -0.25, 1.3], physicsClientId=self.id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)
        self.viewMatrix = p.computeViewMatrixFromYawPitchRoll([0, 0, 1.2], .3, 180, -10, 0, 2)
        fov = 60
        aspect = (self.width / 4) / (self.height / 3)
        nearPlane = 0.01
        farPlane = 100
        self.projMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane)
        self.goal = self.target_pos[self.target_index].copy()

        self.lever_angles = [p.getJointStates(switch, jointIndices=[0], physicsClientId=self.id)[0][0]
                             for switch in self.switches]
        self.curr_step = 0
        self.goal_positions = np.array(self.target_pos).copy()

        return self._get_obs([0])

    def init_start_pos(self):
        """exchange this function for curriculum"""
        self.init_pos = np.array([0, -.5, 1.1])
        if self.stochastic:
            self.init_pos[0] += self.init_pos_random.uniform(-0.5, 0.5)
            self.init_pos[1] += self.init_pos_random.uniform(-0.1, 0.1)
            self.init_pos[2] += self.init_pos_random.uniform(-0.1, 0.1)

    def init_robot_arm(self):
        self.init_start_pos()
        init_orient = p.getQuaternionFromEuler(np.array([0, np.pi / 2.0, 0]), physicsClientId=self.id)
        self.util.ik_random_restarts(self.robot, 11, self.init_pos, init_orient, self.world_creation,
                                     self.robot_left_arm_joint_indices, self.robot_lower_limits,
                                     self.robot_upper_limits,
                                     ik_indices=[0, 1, 2, 3, 4, 5, 6], max_iterations=100, max_ik_random_restarts=10,
                                     random_restart_threshold=0.03, step_sim=True)

        self.world_creation.set_gripper_open_position(self.robot, position=1, left=True, set_instantly=True)
        self.tool = self.world_creation.init_tool(self.robot, mesh_scale=[0.001] * 3, pos_offset=[0, 0, 0.02],
                                                  orient_offset=p.getQuaternionFromEuler([0, -np.pi / 2.0, 0],
                                                                                         physicsClientId=self.id),
                                                  maximal=False)

    def get_random_target(self):
        targets = self.target_pos
        return targets[np.random.randint(self.num_targets)]

    def set_target_index(self, index=None):
        if index is None:
            self.target_index = self.np_random.choice(self.target_indices)
        else:
            self.target_index = index
        self.unique_index = self.target_index

    def reset_noise(self):
        # default wall offset (for pretraining) is randomly chosen between (-0.1, 0)
        # calibration offset should be 0.1, online should be 0
        offset = self.np_random.choice((0.1, 0)) if self.wall_offset is None else self.wall_offset
        self.switch_pos_noise = [self.np_random.uniform(-.25, .05), 0, 0]
        self.wall_noise = [0, self.np_random.uniform(-.05, .05) + offset, 0]

    def calibrate_mode(self, calibrate, split):
        self.wall_offset = 0.1 if split else 0
        self.wall_color = [255 / 255, 187 / 255, 120 / 255, 1] if calibrate else None

    def generate_target(self):
        # Place a switch on a wall
        wall_index = 0
        wall1 = np.array([0., -1., 1.])
        if self.stochastic:
            wall1 = wall1 + self.wall_noise
        walls = [
            (wall1, [0, 0, 0, 1]),
            (np.array([.65, -.4, 1]), p.getQuaternionFromEuler([0, 0, np.pi / 2])),
        ]
        wall_pos, wall_orient = walls[wall_index]
        wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1, .1, 1])
        wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[1, .1, 1], rgbaColor=self.wall_color)
        self.wall = p.createMultiBody(basePosition=wall_pos, baseOrientation=wall_orient,
                                      baseCollisionShapeIndex=wall_collision, baseVisualShapeIndex=wall_visual,
                                      physicsClientId=self.id)

        self.target_string = np.array(self.messages[self.target_index].split(' ')).astype(int)
        self.initial_string = np.ones(self.num_targets)
        self.current_string = self.initial_string.copy()
        wall_pos, wall_orient = p.getBasePositionAndOrientation(self.wall, physicsClientId=self.id)

        switch_spacing = .88 / (self.num_targets - 1)
        switch_center = np.array([-switch_spacing * (len(self.target_string) // 2), .1, 0])
        if self.stochastic:
            switch_center = switch_center + self.switch_pos_noise
        switch_scale = .075
        self.switches = []
        for increment, on_off in zip(np.linspace(np.zeros(3), [switch_spacing * (len(self.target_string) - 1), 0, 0],
                                                 num=len(self.target_string)), self.initial_string):
            switch_pos, switch_orient = p.multiplyTransforms(wall_pos, wall_orient, switch_center + increment,
                                                             p.getQuaternionFromEuler([0, 0, 0]),
                                                             physicsClientId=self.id)
            switch = p.loadURDF(os.path.join(self.world_creation.directory, 'light_switch', 'switch.urdf'),
                                basePosition=switch_pos, useFixedBase=True, baseOrientation=switch_orient, \
                                physicsClientId=self.id, globalScaling=switch_scale)
            self.switches.append(switch)
            p.setCollisionFilterPair(switch, switch, 0, -1, 0, physicsClientId=self.id)
            p.setCollisionFilterPair(switch, self.wall, 0, -1, 0, physicsClientId=self.id)
            p.setCollisionFilterPair(switch, self.wall, -1, -1, 0, physicsClientId=self.id)
            if not on_off:
                p.resetJointState(switch, jointIndex=0, targetValue=LOW_LIMIT, physicsClientId=self.id)
            else:
                p.resetJointState(switch, jointIndex=0, targetValue=HIGH_LIMIT, physicsClientId=self.id)

        sphere_collision = -1
        sphere_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self.success_dist * 1.5,
                                            rgbaColor=[0, 0, 1, 1], physicsClientId=self.id)

        self.targets = []
        self.targets1 = []
        for i, switch in enumerate(self.switches):
            self.targets.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                  basePosition=[-10, -10, -10], useMaximalCoordinates=False,
                                                  physicsClientId=self.id))
            if i == self.target_index:
                self.targets1.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                       baseVisualShapeIndex=sphere_visual, basePosition=[-10, -10, -10],
                                                       useMaximalCoordinates=False, physicsClientId=self.id))
            else:
                self.targets1.append(p.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=sphere_collision,
                                                       basePosition=[-10, -10, -10],
                                                       useMaximalCoordinates=False, physicsClientId=self.id))

        self.update_targets()

    def update_targets(self):
        self.target_pos = []
        self.target_pos1 = []
        for i, switch in enumerate(self.switches):
            switch_pos, switch_orient = p.getLinkState(switch, 0, computeForwardKinematics=True,
                                                       physicsClientId=self.id)[:2]
            lever_pos = np.array([0, .07, .035])
            if self.target_string[i] == 0:
                second_pos = lever_pos + np.array([0, .03, .1])
                target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
                target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
                self.target_pos.append(target_pos)
                self.target_pos1.append(target_pos1)
            else:
                second_pos = lever_pos + np.array([0, .03, -.1])
                target_pos = np.array(p.multiplyTransforms(switch_pos, switch_orient, lever_pos, [0, 0, 0, 1])[0])
                target_pos1 = np.array(p.multiplyTransforms(switch_pos, switch_orient, second_pos, [0, 0, 0, 1])[0])
                self.target_pos.append(target_pos)
                self.target_pos1.append(target_pos1)

            p.resetBasePositionAndOrientation(self.targets[i], target_pos, [0, 0, 0, 1], physicsClientId=self.id)
            p.resetBasePositionAndOrientation(self.targets1[i], target_pos1, [0, 0, 0, 1], physicsClientId=self.id)

    @property
    def tool_pos(self):
        return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[0])

    @property
    def tool_orient(self):
        return np.array(p.getLinkState(self.tool, 1, computeForwardKinematics=True, physicsClientId=self.id)[1])


class OneSwitchJacoEnv(LightSwitchEnv):
    def __init__(self, **kwargs):
        super().__init__(robot_type='jaco', **kwargs)


class ThreeSwitchJacoEnv(LightSwitchEnv):
    def __init__(self, **kwargs):
        super().__init__(num_messages=[0, 1, 2], robot_type='jaco', **kwargs)

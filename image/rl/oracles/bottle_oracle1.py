import pybullet as p
import numpy as np
from collections import deque
from numpy.linalg import norm
from .base_oracles import UserModelOracle

class BottleOracle(UserModelOracle):
	def _query(self,obs,info):
		target = info['target_pos']
		tool_pos = obs['raw_obs'][:3]
		door_pos = obs['raw_obs'][7:10]
		shelf_pos = info['shelf_pos']
		final_door_pos = (np.array([-.15,.17,0]) if info['target_index']//2 else np.array([.15,.17,0])) + shelf_pos
		door_offset = np.array([.02,0,0]) if info['target_index']%2 else np.array([-.02,0,0])
		aux_pos = target + np.array([.1,.4,0])

		if not info['door_open']:
			if not self.aux_reached and info['target_index'] == 3:
				target_pos = aux_pos
			else:
				target_pos = door_pos + door_offset
			if norm(tool_pos-aux_pos) < .05:
				self.aux_reached = True
		else:
			if norm(tool_pos-target) > .2:
				target_pos = target + np.array([0,.2,0])
			else:
				target_pos = target

		old_traj = target_pos - info['old_tool_pos']
		new_traj = info['tool_pos'] - info['old_tool_pos']
		cos_error = np.dot(old_traj,new_traj)/(norm(old_traj)*norm(new_traj))
		criterion = cos_error < self.threshold
		return criterion, target_pos
	def reset(self):
		self.aux_reached = False
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SparseReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)

    def _step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward = 0
        if np.linalg.norm(vec) < 0.04: # custom
            reward = 1
        self.do_simulation(a, self.frame_skip)
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        #qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos = self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            #self.goal = [0.21, 0.04] #[0.2, 0.01]
            if np.linalg.norm(self.goal) < 2:
                break
        #self.goal = [0.2, 0.01]
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        #print (self.get_body_com("fingertip"))
        #print (self.get_body_com("target"))
        #dist = self.get_body_com("fingertip")-self.get_body_com("target")
        #print (np.linalg.norm(dist))
        return self._get_obs()

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[2:],
            self.model.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])

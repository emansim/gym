import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class GripperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, '3link_gripper_pusher.xml', 2)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 5.0
        #self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        # set position of object and goal
        while True:
            self.goal = self.np_random.uniform(low=-1.2, high=1.2, size=2)
            self.object = self.np_random.uniform(low=-1.2, high=1.2, size=2)
            if np.linalg.norm(self.goal) > 0.2 and np.linalg.norm(self.object) > 0.2:
                break

        qpos[-2:] = self.object
        qpos[-4:-2] = self.goal

        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        vec = self.get_body_com("object")-self.get_body_com("goal")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:4]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[4:],
            self.model.data.qvel.flat[:4],
            self.get_body_com("object")-self.get_body_com("goal")
        ])
'''
if __name__ == "__main__":
    import time
    env = GripperEnv()
    env.reset()

    while True:
        env.reset()
        for _ in range(150):
            env.render()
            action = np.random.uniform(low=-2,high=2,size=env.action_space.shape)
            #action[-1] = 0
            env.step(action)
        time.sleep(1)
'''

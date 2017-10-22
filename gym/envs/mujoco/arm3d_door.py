import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class Arm3dDoorEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.target_doorpos = np.asarray([-0.08406698, -0.10218753, 0.3])
        self.target_handlepos = np.asarray([-0.01953642, -0.41961081, -0.25])
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'pr2_arm3d_door.xml', 2)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = self.model.stat.extent * 2.0
        #self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = 0.

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _step(self, a):
        vec1 = self.get_body_com("r_gripper_l_finger_tip_link")-self.get_body_com("door_handle")
        vec2 = self.get_body_com("door_handle")-self.target_handlepos
        reward_dist = - np.linalg.norm(vec1+vec2)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        theta = self.model.data.qpos.flat[:6]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[6:],
            self.model.data.qvel.flat[:6],
            self.get_body_com("r_gripper_l_finger_tip_link")-self.get_body_com("door_handle"),
            self.get_body_com("door_handle")-self.target_handlepos

        ])

'''
#0.16134199 -0.37169818  0.25

#-0.01953642 -0.41961081 door handle pos
# -0.08406698 -0.10218753 door pos
if __name__ == "__main__":
    env = Arm3dDoor()
    print (env.model.data.qpos.flat[:].shape)
    print (env.action_space)
    """
    import time
    env = Arm3dDoor()
    env.reset()

    print (env.action_space)
    episode_i = 0
    """

    while True:
        env.reset()
        env.render()
        """
        print ("episode {}".format(episode_i))
        print ('orig')
        print (env.get_body_com("door_handle"))
        print (env.get_body_com("door"))

        for _ in range(150):
            env.render()
            #action = np.random.uniform(low=-2,high=2,size=env.action_space.shape)
            action = np.zeros(env.action_space.shape)
            #action[-1] = 0.
            #action[-1] = 0
            env.step(action)
        print ('end')
        print (env.get_body_com("door_handle"))
        print (env.get_body_com("door"))

        episode_i += 1
        time.sleep(1)
        """
'''

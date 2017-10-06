import gym
import numpy as np
import time
env=gym.make('SparseReacher-v1')
env.reset()
for i in range(50):
    env.render()
    _, reward, _, _ = env.step(np.random.uniform(low=-1,high=1,size=env.action_space.shape))
    print ("reward {}".format(reward))
    time.sleep(0.1)

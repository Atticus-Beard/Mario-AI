from environment import MarioEnvironment
import time
import random

env = MarioEnvironment()

observation = env.reset()

done = False

while not done:

    grid = env.get_grid()
    print(grid)

    action = 2
    observation, reward, done = env.step(action)

    print(
        "Reward:",
        reward,
        "Observation:",
        observation.shape
    )

    time.sleep(1/60)

env.close()
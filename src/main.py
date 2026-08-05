from environment import MarioEnvironment
import time
import random

env = MarioEnvironment()

observation = env.reset()

done = False

while not done:
    action = 2  # NOOP
    observation, reward, done = env.step(action)

    print(
        "Reward:",
        reward,
        "Observation:",
        observation.shape
    )

    time.sleep(1/60)

env.close()
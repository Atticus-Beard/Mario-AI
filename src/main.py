from environment import MarioEnvironment
from fitness import Fitness
import time
import random

env = MarioEnvironment()
fitness = Fitness()

observation = env.reset()

done = False

while not done:

    grid = env.get_grid()
    print(grid)

    action = random.randint(0, 5)
    observation, reward, done, info = env.step(action)
    score = fitness.update(info, done)

    print(f"Score: {score}.")

    time.sleep(1/60)

env.reset()
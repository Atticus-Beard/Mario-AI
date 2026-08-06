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
    player_state = env.get_player_state()
    print(grid)

    action = random.randint(0, 5)
    observation, reward, done, info = env.step(action)
    score, stuck = fitness.update(info, done, player_state)

    print(
        f"Score: {score:.2f} | "
        f"X: {info['x_pos']} | "
        f"State: {hex(player_state)}"
    )


    # Stop this Mario if it gets stuck
    if stuck:
        print("Mario got stuck. Ending run.")
        done = True

    time.sleep(1/60)

print("Final fitness:", fitness.score)

env.close()
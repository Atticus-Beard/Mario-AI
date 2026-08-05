from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

class MarioEnvironment:
    def __init__(self):
        env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human")

        # When using COMPLEX_MOVEMENT there is 12 discrete actions available. The actions are:
        # 0. NOOP
        # 1. Right
        # 2. Right + A
        # 3. Right + B
        # 4. Right + A + B
        # 5. Left
        # 6. Left + A
        # 7. Left + B
        # 8. Left + A + B
        # 9. Down
        # 10. Up

        # When using SIMPLE_MOVEMENT there is 7 discrete actions available. The actions are:
        # 0. NOOP
        # 1. Right
        # 2. Right + A
        # 3. Right + B
        # 4. Right + A + B
        # 5. A
        # 6. Left 

        # We are going to use COMPLEX_MOVEMENT for this project, to give the agent more options to move around the environment.
        # You can refer to the documentation for more information on the different action spaces available in gym-super-mario-bros.  
        self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.env.render()
        return observation, reward, done

    def close(self):
        self.env.close()
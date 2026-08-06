from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
import numpy as np

class MarioEnvironment:
    def __init__(self):
        env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode="human")

        # When using COMPLEX_MOVEMENT there is 12 discrete actions available. The actions are:
        # 0. NOOP
        # 1. Right
        # 2. Right + A
        # 3. Right + B
        # 4. Right + A + B
        # 5. A
        # 6. Left
        # 7. Left + A
        # 8. Left + B
        # 9. Left + A + B 
        # 10. Down
        # 11. Up

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

        # We are going to use the RAM state of the environment to get information about the game state, such as the position of Mario and the enemies. 
        # The RAM state is a 2048-byte array that contains the current state of the game. We can access the RAM state using the `ram` attribute of the environment.
        self.ram = self.env.unwrapped.ram

    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.env.render()
        return observation, reward, done, info

    def close(self):
        self.env.close()

    
    def tile_loc_to_ram_address(self, x, y):
        page = x // 16
        x_loc = x % 16

        y_loc = page * 13 + y

        address = 0x500 + x_loc + y_loc * 16

        return address


    def get_grid(self):
        width = 16
        height = 13

        grid = np.zeros((height, width))


        # Mario position
        mario_level_x = (
            int(self.ram[0x6D]) * 256
            + int(self.ram[0x86])
        )

        mario_x = int(self.ram[0x3AD])
        mario_y = int(self.ram[0x3B8]) + 16


        camera_x = mario_level_x - mario_x

        screen_start = int(camera_x / 16)


        # Ground / blocks
        for x in range(width):
            for y in range(height):

                tile_x = (screen_start + x) % 32
                tile_y = y

                address = self.tile_loc_to_ram_address(
                    tile_x,
                    tile_y
                )

                if self.ram[address] != 0:
                    title = self.ram[address]
                    grid[y][x] = self.get_tile_type(title)

        # Mario
        mario_grid_x = (mario_x + 8) // 16
        mario_grid_y = (mario_y - 32) // 16

        if (
            0 <= mario_grid_x < width
            and 0 <= mario_grid_y < height
        ):
            grid[mario_grid_y][mario_grid_x] = 2


        # Enemies
        for i in range(5):

            if self.ram[0xF + i] == 1:

                enemy_x = (
                    int(self.ram[0x6E + i]) * 256
                    + int(self.ram[0x87 + i])
                    - camera_x
                )

                enemy_y = int(self.ram[0xCF + i])


                enemy_grid_x = (enemy_x + 8) // 16
                enemy_grid_y = (enemy_y + 8 - 32) // 16


                if (
                    0 <= enemy_grid_x < width
                    and 0 <= enemy_grid_y < height
                ):
                    grid[enemy_grid_y][enemy_grid_x] = 3


        return grid

    def get_tile_type(self, tile):
        # Empty
        if tile == 0:
            return 0

        # Question blocks
        if tile in [0xC0, 0xC1]:
            return 4

        # Brick blocks
        if tile in [0x01, 0x02]:
            return 5

        # Pipes
        if tile in [0x0A, 0x0B, 0x0C, 0x0D]:
            return 6

        # Everything else solid
        return 1
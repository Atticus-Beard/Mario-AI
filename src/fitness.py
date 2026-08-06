class Fitness:
    def __init__(self):
        self.best_x = 0
        self.score = 0

    def update(self, info, done):
        mario_x = info["x_pos"]

        if mario_x > self.best_x:
            self.score += mario_x - self.best_x
            self.best_x = mario_x

        if done:
            self.score -= 500

        return self.score
class Fitness:
    def __init__(self):
        self.score = 0

        # Furthest X position Mario has reached
        self.best_x = 0

        # Previous values so we can dectet changes.
        self.previous_coins = 0 
        self.previous_score = 0 
        self.previous_life = 2
        self.previous_status = "small"

        self.previous_world = 1
        self.previous_stage = 1
        self.in_cutscene = False

        self.previous_player_state = 0x08

        self.frames_without_progress = 0
        self.max_frames_without_progress = 900

        self.stuck = False

    def update(self, info, done, player_state):
        state_changed = player_state != self.previous_player_state

        # --------------------------
        # Player dies.
        # --------------------------

        if state_changed and player_state in (0x06, 0x0B):
            self.score -= 500

        # --------------------------
        # Level completed
        # --------------------------

        if(
            info["world"] != self.previous_world or
            info["stage"] != self.previous_stage
        ):
            self.score += 5000

            self.previous_world = info["world"]
            self.previous_stage = info["stage"]
            self.frames_without_progress = 0

            # Reset the best_x for the new level as SMB resets coords for each level.
            self.best_x = 0

        self.previous_player_state = player_state

        # Pause all other fitnes calculations. (So we do not reward/punish the agent for being in a cutscene.)
        if player_state != 0x08:
            return self.score, self.stuck

        # Everything below is only normal gameplay.
        mario_x = info["x_pos"]

        # Reward a tiny amount for surving each frame. 
        self.score += 0.01

        # --------------------------
        # Reward moving to the right
        # --------------------------

        if mario_x > self.best_x:
            self.score += mario_x - self.best_x
            self.best_x = mario_x
            self.frames_without_progress = 0
        else:
            self.frames_without_progress += 1


        # --------------------------
        # End the agents run
        # if it sits idle too long.
        # --------------------------
        if self.frames_without_progress > self.max_frames_without_progress:
            self.stuck = True

        # --------------------------
        # Reward coins
        # --------------------------

        if info["coins"] > self.previous_coins:
            gained = info["coins"] - self.previous_coins
            self.score += gained * 100
            self.previous_coins = info["coins"]

        # --------------------------
        # Reward game score
        # (enemy kills, blocks, etc)
        # --------------------------

        if info["score"] > self.previous_score:
            gained = info["score"] - self.previous_score
            self.score += gained * 0.1
            self.previous_score = info["score"]

        # --------------------------
        # Reward powerups
        # and penalize loosing one.
        # --------------------------

        if info["status"] != self.previous_status:

            # Punish if mario gets hit 
            if self.previous_status == "fireball" and info["status"] == "tall":
                self.score -= 250

            elif self.previous_status == "tall" and info["status"] == "small":
                self.score -= 500
    
            # Reward upgraded powerup
            elif info["status"] == "tall":
                self.score += 500

            elif info["status"] == "fireball":
                self.score += 1000

            self.previous_status = info["status"]

        return self.score, self.stuck
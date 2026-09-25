class Fitness:
    def __init__(self):
        self.score = 0

        # Furthest X position Mario has reached
        self.best_x = 0

        # Previous values so we detect changes
        self.previous_coins = 0
        self.previous_score = 0
        self.previous_life = 2
        self.previous_status = "small"

        self.previous_world = 1
        self.previous_stage = 1

        self.previous_player_state = 0x08

        self.frames_without_progress = 0
        self.max_frames_without_progress = 300

        self.stuck = False


    def update(self, info, done, player_state):

        state_changed = (
            player_state != self.previous_player_state
        )


        # --------------------------
        # Player dies
        # --------------------------

        if state_changed and player_state in (0x06, 0x0B):
            self.score -= 1000


        # --------------------------
        # Level completed
        # --------------------------

        if (
            info["world"] != self.previous_world or
            info["stage"] != self.previous_stage
        ):

            self.score += 5000

            self.previous_world = info["world"]
            self.previous_stage = info["stage"]

            self.frames_without_progress = 0

            # Reset x because Mario coordinates reset
            self.best_x = 0


        self.previous_player_state = player_state


        # Pause scoring during cutscenes
        if player_state != 0x08:
            return self.score, self.stuck



        mario_x = info["x_pos"]


        # --------------------------
        # Reward moving right
        # --------------------------


        if mario_x > self.best_x:

            distance = mario_x - self.best_x

            self.score += distance * 0.2

            self.best_x = mario_x

            self.frames_without_progress = 0

        else:

            self.frames_without_progress += 1

            if self.frames_without_progress > 30:
                self.score -= 0.1 * (
                    self.frames_without_progress / 30
                )


        # --------------------------
        # Stop if stuck
        # --------------------------

        if (
            self.frames_without_progress >
            self.max_frames_without_progress
        ):
            self.stuck = True



        # --------------------------
        # Reward coins
        # --------------------------

        if info["coins"] > self.previous_coins:

            gained = (
                info["coins"] -
                self.previous_coins
            )

            self.score += gained * 100

            self.previous_coins = info["coins"]



        # --------------------------
        # Reward game score
        # --------------------------

        if info["score"] > self.previous_score:

            gained = (
                info["score"] -
                self.previous_score
            )

            self.score += gained * 0.05

            self.previous_score = info["score"]



        # --------------------------
        # Powerups
        # --------------------------

        if info["status"] != self.previous_status:


            # Lost fire power
            if (
                self.previous_status == "fireball"
                and info["status"] == "tall"
            ):
                self.score -= 250


            # Lost size
            elif (
                self.previous_status == "tall"
                and info["status"] == "small"
            ):
                self.score -= 1000


            # Upgrade
            elif info["status"] == "tall":

                self.score += 500


            elif info["status"] == "fireball":

                self.score += 1000


            self.previous_status = info["status"]


        return self.score, self.stuck
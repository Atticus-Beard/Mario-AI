from environment import MarioEnvironment
from fitness import Fitness
from model import NeuralNetwork

from pathlib import Path
import numpy as np


SAVE_PATH = Path("saves/best_model.pkl")
SAVE_PATH.parent.mkdir(exist_ok=True)


class Trainer:

    def __init__(self, population_size=20):

        self.population_size = population_size

        self.population = [
            NeuralNetwork(
                [
                    16 * 13,  # Grid input
                    64,
                    32,
                    12       # Actions
                ]
            )
            for _ in range(population_size)
        ]

        self.generation = 1


    def create_games(self):

        self.games = [
            MarioEnvironment(render=False)
            for _ in range(self.population_size)
        ]

        # IMPORTANT:
        # Gym requires reset before step()
        for env in self.games:
            env.reset()


        self.fitness = [
            Fitness()
            for _ in range(self.population_size)
        ]


    def run_generation(self):

        print(
            f"\nGeneration {self.generation}"
        )


        self.create_games()


        done = [
            False
            for _ in range(self.population_size)
        ]


        frames = 0


        while not all(done):

            for i in range(self.population_size):

                # Skip dead Marios
                if done[i]:
                    continue


                env = self.games[i]
                brain = self.population[i]
                fitness = self.fitness[i]


                # Get the world as a grid
                grid = env.get_grid()


                # Neural network chooses action
                action = brain.choose_action(grid)


                observation, reward, finished, info = env.step(action)


                player_state = env.get_player_state()


                score, stuck = fitness.update(
                    info,
                    finished,
                    player_state
                )


                # End this Mario only
                if stuck:
                    finished = True


                done[i] = finished


                brain.fitness = score


                # Don't print every frame
                if frames % 60 == 0:

                    print(
                        f"Mario {i} | "
                        f"Fitness: {score:.2f} | "
                        f"X: {info['x_pos']}"
                    )


            frames += 1


        self.finish_generation()



    def finish_generation(self):

        print("\nGeneration finished")


        # Highest fitness first
        self.population.sort(
            key=lambda x: x.fitness,
            reverse=True
        )


        best = self.population[0]


        print(
            f"Best fitness: {best.fitness:.2f}"
        )


        # Save best brain
        best.save(
            SAVE_PATH
        )


        print(
            "Saved best model"
        )


        #
        # Create next generation
        #

        new_population = []


        # Keep winner
        new_population.append(
            best.copy()
        )


        while len(new_population) < self.population_size:

            child = best.copy()

            child.mutate(
                mutation_rate=0.05,
                mutation_strength=0.30
            )

            new_population.append(child)


        self.population = new_population


        # Close old games
        for env in self.games:
            env.close()


        self.generation += 1



    def train(self):

        while True:

            self.run_generation()
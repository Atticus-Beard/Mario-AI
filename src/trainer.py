from environment import MarioEnvironment
from fitness import Fitness
from model import NeuralNetwork

from multiprocessing import Pool, cpu_count

from pathlib import Path
import numpy as np
import csv
import pickle
import time


SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

LOG_FILE = SAVE_DIR / "fitness.csv"


if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "best_fitness",
            "average_fitness"
        ])


def init_worker():
    global worker_env

    worker_env = MarioEnvironment(render=False)

    print("Worker ready.")


def evaluate_brain(brain):
    global worker_env

    env = worker_env
    fitness = Fitness()

    env.reset()

    done = False
    score = 0

    while not done:

        grid = env.get_grid()

        action = brain.choose_action(grid)

        observation, reward, finished, info = env.step(action)

        player_state = env.get_player_state()

        score, stuck = fitness.update(
            info,
            finished,
            player_state
        )

        if stuck:
            done = True
        else:
            done = finished


    brain.fitness = score

    return brain, score



class Trainer:
    
    def __init__(
        self,
        population_size=20,
        generation=1,
        population=None,
        best_fitness=-1
    ):

        self.population_size = population_size
        self.generation = generation
        self.best_fitness = best_fitness

        if population is not None:

            print("Loading saved population.")

            self.population = population

        else:

            print("Creating new population.")

            self.population = [
                NeuralNetwork(
                    [
                        16 * 13,
                        64,
                        32,
                        12
                    ]
                )
                for _ in range(population_size)
            ]

        workers = min(
            cpu_count(),
            self.population_size
        )

        print(
            f"Starting {workers} CPU workers..."
        )

        self.pool = Pool(
            workers,
            initializer=init_worker,
            maxtasksperchild=None
        )

    def run_generation(self):

        print(
            f"\nGeneration {self.generation}"
        )

        self.start = time.time()


        results = self.pool.map(
            evaluate_brain,
            self.population
        )


        self.population = []


        for brain, score in results:

            print(
                f"Fitness: {score:.2f}"
            )

            self.population.append(brain)



        self.finish_generation()



    def finish_generation(self):

        mutation_rate = max(
            0.02,
            0.15 * (0.995 ** self.generation)
        )


        print("\nGeneration finished")

        generation_time = time.time() - self.start

        evaluations_per_second = (
           self.population_size / generation_time
        )

        print(
            f"Generation time: {generation_time:.2f} seconds."
        )

        print(
            f"Generations/hour: {3600 / generation_time:.2f}"
        )

        print(
            f"Evaluations/sec: {evaluations_per_second:.2f}"
        )


        self.population.sort(
            key=lambda x: x.fitness,
            reverse=True
        )


        best = self.population[0]


        average = np.mean(
            [
                brain.fitness
                for brain in self.population
            ]
        )


        # SAVE FITNESS CSV
        with open(LOG_FILE, "a", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(
                [
                    self.generation,
                    best.fitness,
                    average
                ]
            )


        print(
            f"Average fitness: {average:.2f}"
        )

        print(
            f"Best fitness: {best.fitness:.2f}"
        )



        generation_path = (
            SAVE_DIR /
            f"population_{self.generation}.pkl"
        )


        with open(generation_path, "wb") as f:

            pickle.dump(
                {
                    "generation": self.generation,
                    "population": self.population,
                    "best_fitness": best.fitness
                },
                f
            )


        print(
            "Saved entire population"
        )


        if best.fitness > self.best_fitness:

            self.best_fitness = best.fitness


            best.save(
                SAVE_DIR / "best_model.pkl",
                generation=self.generation,
                fitness=best.fitness
            )


            print(
                "New best model saved!!!"
            )

        # Cleanup old population saves.
        self.cleanup_old_populations()


        #
        # Create next generation
        #

        new_population = []


        elite_count = min(
            5,
            self.population_size
        )


        breeding_pool = min(
            30,
            self.population_size
        )


        random_count = max(
            10,
            int(self.population_size * 0.10)
        )


        target_size = (
            self.population_size -
            random_count
        )


        fitnesses = np.array(
            [
                max(brain.fitness, 0)
                + 1
                + np.random.random()

                for brain in self.population[:breeding_pool]
            ]
        )


        probabilities = (
            fitnesses /
            fitnesses.sum()
        )


        # Keep elites

        for i in range(elite_count):

            elite = self.population[i].copy()

            if i == elite_count - 1:

                elite.mutate(
                    mutation_rate=mutation_rate,
                    mutation_strength=0.1
                )


            new_population.append(elite)



        # Breed

        while len(new_population) < target_size:


            p1, p2 = np.random.choice(
                breeding_pool,
                size=2,
                replace=False,
                p=probabilities
            )


            parent1 = self.population[p1]
            parent2 = self.population[p2]


            child = NeuralNetwork.breed(
                parent1,
                parent2
            )


            child.mutate(
                mutation_rate=mutation_rate,
                mutation_strength=np.random.uniform(
                    0.15,
                    0.35
                )
            )


            new_population.append(child)



        # Random brains

        while len(new_population) < self.population_size:

            new_population.append(
                NeuralNetwork(
                    [
                        16 * 13,
                        64,
                        32,
                        12
                    ]
                )
            )


        print(
            f"Added {random_count} random brains"
        )


        self.population = new_population


        self.generation += 1



    def train(self):

        while True:

            self.run_generation()



    def close(self):

        print(
            "Closing workers..."
        )

        self.pool.close()
        self.pool.join()

    def cleanup_old_populations(self):
        populations = list(
            SAVE_DIR.glob("population_*.pkl")
        )

        #Sort by generation number.
        populations.sort(
            key=lambda p: int(p.stem.split("_")[1])
        )

        old_populations = populations[:-250]

        for path in old_populations:
            path.unlink()

        if old_populations:
            print(f"Deleted {len(old_populations)} old population saves.")


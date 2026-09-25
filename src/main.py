from environment import MarioEnvironment
from fitness import Fitness
from model import NeuralNetwork
from trainer import Trainer

from pathlib import Path
import time
import pickle

def get_latest_population():
    populations = list(SAVE_DIR.glob("population_*.pkl"))

    if not populations:
        return None

    # Sort by generation number
    populations.sort(
        key=lambda p: int(p.stem.split("_")[1])
    )

    return populations[-1]


SAVE_PATH = Path("saves/best_model.pkl")
SAVE_DIR = Path("saves")

POPULATION_PATH = get_latest_population()

POPULATION_SIZE = 500

def banner():
    print(r"""
    ,---.    ,---.   ____    .-------.   .-./`)     ,-----.               ____    .-./`)  
    |    \  /    | .'  __ `. |  _ _   \  \ .-.')  .'  .-,  '.           .'  __ `. \ .-.') 
    |  ,  \/  ,  |/   '  \  \| ( ' )  |  / `-' \ / ,-.|  \ _ \         /   '  \  \/ `-' \ 
    |  |\_   /|  ||___|  /  ||(_ o _) /   `-'`"`;  \  '_ /  | :        |___|  /  | `-'`"` 
    |  _( )_/ |  |   _.-`   || (_,_).' __ .---. |  _`,/ \ _/  |           _.-`   | .---.  
    | (_ o _) |  |.'   _    ||  |\ \  |  ||   | : (  '\_/ \   ;        .'   _    | |   |  
    |  (_,_)  |  ||  _( )_  ||  | \ `'   /|   |  \ `"/  \  ) /         |  _( )_  | |   |  
    |  |      |  |\ (_ o _) /|  |  \    / |   |   '. \_/``".'          \ (_ o _) / |   |  
    '--'      '--' '.(_,_).' ''-'   `'-'  '---'     '-----'             '.(_,_).'  '---' 

    Super Mario Bros AI
    Built from scratch                                                                        
    """)

def play_model(model):
    print("Starting Mario...")

    env = MarioEnvironment(
        render=True
    )

    env.reset()

    done = False
    frames = 0

    while not done and frames < 10000:
        grid = env.get_grid()

        action = model.choose_action(
            grid
        )

        observation, reward, done, info = env.step(
            action
        )

        print(
            f"X: {info['x_pos']} | "
            f"World: {info['world']}-{info['stage']}"
        )
        frames += 1

        time.sleep(1/60)

    print("Mario Finished!")
    env.close()


def main():
    banner()

    while True:
        print()
        print("Commands:")
        print("----------------")
        print("start  - Create new AI")
        print("load   - Load saved AI")
        print("play   - Watch AI play")
        print("exit   - Quit")
        print()

        command = input("> ").lower().strip()

        if command == "start":
            print("Creating new neural network...")
            

            trainer = Trainer(
                population_size=POPULATION_SIZE
            )

            try:
                trainer.train()
            finally:
                trainer.close()



        elif command == "load":

            if not SAVE_PATH.exists():

                print(
                    "No saved model found."
                )

                continue


            print(
                "Loading model..."
            )

            population_path = get_latest_population()

            if population_path is None:
                print("No population save found.")
                continue

            with open(POPULATION_PATH, "rb") as f:
                checkpoint = pickle.load(f)

            population = checkpoint["population"]

            generation = checkpoint["generation"]

            print(
                f"Loaded generation {generation}"
            )

            print(
                f"Population size {len(population)}"
            )


            print(
                "Starting training from saved model..."
            )


            trainer = Trainer(
                    population_size=POPULATION_SIZE,
                    generation=checkpoint["generation"] + 1,
                    population=population,
                    best_fitness=checkpoint["best_fitness"]
                )


            trainer.train() 


        elif command == "play":
            print("Loading model...")

            if not SAVE_PATH.exists():
                print(
                    "No saved model found."
                )
                continue

            print(
                "Loading best model..."
            )

            checkpoint = NeuralNetwork.load(
                SAVE_PATH
            )

            model = checkpoint["model"]

            print(
                f"Loaded generation {checkpoint['generation']}"
            )

            print(
                f"Fitness: {checkpoint['fitness']}"
            )

            play_model(model)


        elif command == "exit":

            print("Goodbye!")

            break


        else:
            print("Unknown command.")



if __name__ == "__main__":
    main()
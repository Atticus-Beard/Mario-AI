from environment import MarioEnvironment
from fitness import Fitness
from model import NeuralNetwork
from trainer import Trainer

from pathlib import Path
import time


SAVE_PATH = Path("saves/best_model.pkl")

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

    while not done:
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

        time.sleep(1/60)

    print("Mario Finished!")
    env.close


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
                population_size=50
            )

            trainer.train()



        elif command == "load":
            print("Loading model...")

            if not SAVE_PATH.exists():

                print(
                    "No saved model found."
                )

                continue


            print(
                "Loading model..."
            )


            model = NeuralNetwork.load(
                SAVE_PATH
            )


            print(
                "Model loaded."
            )



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


            model = NeuralNetwork.load(
                SAVE_PATH
            )

            play_model(
                model
            )





        elif command == "exit":

            print("Goodbye!")

            break


        else:
            print("Unknown command.")



if __name__ == "__main__":
    main()
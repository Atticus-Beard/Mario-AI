import pandas as pd
import matplotlib.pyplot as plt
import time


while True:
    data = pd.read_csv("/Users/atticus/Desktop/dev/mario-ai/saves/fitness.csv")

    plt.plot(
        data["generation"],
        data["best_fitness"],
        label="Best Fitness"
    )

    plt.plot(
        data["generation"],
        data["average_fitness"],
        label = "Average fitness"
    )

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Mario AI Training")
    plt.grid(True)
    plt.legend()

    plt.show()

    plt.pause(5)
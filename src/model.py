import numpy as np
import pickle

class NeuralNetwork:
    def __init__(self, layers):

        # We store every layer's weights and biases here.
        #
        # Example:
        # layers = [208, 64, 32, 12]
        #
        # Creates:
        # 208 -> 64
        # 64  -> 32
        # 32  -> 12
        #
        # Each connection has a weight that controls
        # how strongly one neuron affects another.

        self.layers = layers

        #Filled in after a mario run
        self.fitness = 0

        self.weights = []
        self.biases = []

        # Create the brain layer by layer.
        for i in range(len(layers) - 1):

            # Weight matrix between two layers
            self.weights.append(
                np.random.randn(
                    layers[i],
                    layers[i + 1]
                )
            )

            # Every neuron gets a bias.
            # Bias shifts when the neuron activates

            self.biases.append(
                np.random.randn(
                    layers[i +1]
                )
            )

    def activation(self, x):
        """
        Activation function.

        np.tanh keeps outputs between -1 and 1.
        """

        return np.tanh(x)

    def predict(self, inputs):
        output = inputs

        for weight, bias, in zip(
            self.weights,
            self.biases
        ):


            # Multiply inputs by weights.
            # Add the bias.
            # output = input * importance + offset

            output = np.dot(
                output,
                weight
            ) + bias

            # Activate the neuron
            output = self.activation(output)

        return output

    def choose_action(self, grid):
        inputs = grid.flatten()

        output = self.predict(inputs)

        return np.argmax(output)

    def copy(self):
        """
        Creates an identical copy of this brain.
        """

        child = NeuralNetwork(self.layers)

        child.weights = [
            weight.copy()
            for weight in self.weights
        ]

        child.biases = [
            bias.copy()
            for bias in self.biases
        ]

        child.fitness = self.fitness

        return child

    def mutate(
        self,
        mutation_rate=0.05,
        mutation_strength=0.30
    ):
        """
        Randomly change a small percentage of weights
        and biases.
        """

        for weight in self.weights:

            mask = (
                np.random.rand(*weight.shape)
                < mutation_rate
            )

            weight += (
                np.random.randn(*weight.shape)
                * mutation_strength
                * mask
            )

        for bias in self.biases:

            mask = (
                np.random.rand(*bias.shape)
                < mutation_rate
            )

            bias += (
                np.random.randn(*bias.shape)
                * mutation_strength
                * mask
            )

    @staticmethod
    def breed(parent1, parent2):
        """
        Create a child by combining two parents.
        Both parents must have the same architecture.
        """

        child = NeuralNetwork(parent1.layers)

        child.weights = []
        child.biases = []

        for w1, w2 in zip(parent1.weights, parent2.weights):

            mask = np.random.rand(*w1.shape) < 0.5

            child.weights.append(
                np.where(mask, w1, w2)
            )

        for b1, b2 in zip(parent1.biases, parent2.biases):

            mask = np.random.rand(*b1.shape) < 0.5

            child.biases.append(
                np.where(mask, b1, b2)
            )

        child.fitness = 0

        return child



    
    def save(self, filename, generation=0, fitness=0):

        checkpoint = {
            "model": self,
            "generation": generation,
            "fitness": fitness
        }

        with open(filename, "wb") as file:
            pickle.dump(checkpoint, file)

    @staticmethod
    def load(filename):

        with open(filename, "rb") as file:
            checkpoint = pickle.load(file)

        return checkpoint
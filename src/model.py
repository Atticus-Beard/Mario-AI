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

        child.fitness = 0

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

    def save(self, filename):
        """
        Saves the neural network to disk.
        """

        with open(filename, "wb") as file:
            pickle.dump(self, file=file)

    @staticmethod
    def load(filename):
        """
        Load a saved neural network.
        """

        with open(filename, "rb") as file:
            return pickle.load(file)

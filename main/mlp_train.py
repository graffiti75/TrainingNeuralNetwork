import numpy as np

# 1. Save activations and derivatives
# 2. Implement BackPropagation
# 3. Implement Gradient Descent
# 4. Implement training
# 5. Train our net with some dummy dataset
# 6. Make some predictions

class MLP:
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs = 3, hidden_layers = [3, 3], num_outputs = 2):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs

        # Create a generic representation of the layers
        layers = [self.num_inputs] + self.hidden_layers + [self.num_outputs]

        # Initiate random weights
        # Create random connection weights for the layers
        self.weights = []
        for i in range(len(layers) - 1):
            w = np.random.rand(layers[i], layers[i + 1])
            # print("-------------------------")
            # print("i: {}".format(i))
            # print("layers[i]: {}".format(layers[i]))
            # print("layers[i + 1]: {}".format(layers[i + 1]))
            # print("layers: {}".format(layers))
            # print("w: {}".format(w))
            # print("-------------------------")
            self.weights.append(w)

        # List of arrays where each array in the list represents the activations for a given layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

        derivatives = []
        # Derivatives must have the same number as the weight's number
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

    def forward_propagate(self, inputs):
        """Computes forward propagation of the network based on input signals.

        Args:
            inputs (ndarray): Input signals
        Returns:
            activations (ndarray): Output values
        """

        # The input layer activation is just the input itself
        activations = inputs
        self.derivatives[0] = inputs

        # Iterate through the network layers
        for i, w in enumerate(self.weights):
            # Calculate matrix multiplication between previous activation and weight matrix
            # Calculate net inputs
            net_inputs = np.dot(activations, w)

            # Apply sigmoid activation function
            # Calculate the activations
            activations = self._sigmoid(net_inputs)     # a_3 = sigmoid(h_3)
            self.activations[i + 1] = activations       # h_3 = a_2 * W_2

        # Return output layer activation
        return activations

    def back_propagate(self, error, verbose = False):
        # x = input layer, W = weights between each neuron layer
        # h_2 = x W_2
        # a_2 = f(h_2)
        #
        # h_3 = a_2 W_2
        # y = f(h_3)
        #
        # y = The outcome we are expecting
        # a = The prediction we got
        #
        # dE/dW_i = (y - a_[i + 1]) . σ'(h_[i + 1])) . a_i
        # dE/dW_i = error . σ'(h_[i + 1]) . a_i
        # σ'(h_[i + 1]) = σ(h_[i + 1])(1 - σ(h_[i + 1]))
        # σ(h_[i + 1]) = a_[i + 1]

        # dE/dW_[i - 1] = (y - a_[i + 1]) . σ'(h_[i + 1])) . W_i . σ'(h_i) . a_[i - 1]

        for i in reversed(range(len(self.derivatives))):
            # activations = h_[i + 1]
            activations = self.activations[i + 1]
            # error = (y - a_[i + 1])
            delta = error * self._sigmoid_derivative(activations) # ndarray([0.1, 0.2]) -> ndarray([[0.1, 0.2]])
            delta_reshaped = delta.reshape(delta.shape[0], -1).T
            current_activations = self.activations[i] # ndarray([0.1, 0.2]) -> ndarray([[0.1], [0.2]])
            current_activations_reshaped = current_activations.reshape(current_activations.shape[0], -1)
            self.derivatives[i] = np.dot(current_activations_reshaped, delta_reshaped)
            # error = (y - a_[i + 1]) . σ'(h_[i + 1])) . W_i
            error = np.dot(delta, self.weights[i].T)

            if verbose:
                print("Derivatives for W{}: {}".format(i, self.derivatives[i]))

        return error

    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            weights = self.weights[i]
            print("Original W{} {}".format(i, weights))
            derivatives = self.derivatives[i]
            weights += derivatives * learning_rate
            print("Updated W{} {}".format(i, weights))

    def _sigmoid_derivative(self, x):
        return x * (1.0 - x)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    # Create a MPL (2 inputs, 1 hidden layer with 5 neurons, 1 output layer with 1 neuron)
    mlp = MLP(2, [5], 1)

    # Create dummy data
    input = np.array([0.1, 0.2])
    target = np.array([0.3])

    # Perform forward propagation
    output = mlp.forward_propagate(input)

    # Calculate error
    error = target - output

    # Perform back propagation
    mlp.back_propagate(error)

    # Apply gradient descent
    mlp.gradient_descent(learning_rate = 0.1)

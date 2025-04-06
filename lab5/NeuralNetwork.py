import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

with open("config.json", "r") as file:
    config = json.load(file)


file_path = config["data"]["file_path"]
SEED = config["data"]["random_seed"]
np.random.seed(SEED)


# Load data

df = pd.read_csv(file_path)

X = df.drop(columns=["quality"]).values
y = df["quality"].values
N, D = X.shape

y_one_hot = np.zeros((N, len(np.unique(y))))
for i, label in enumerate(y):
    # 3-9 -> 0-6
    # 4 -> [0, 1, 0, 0, 0, 0, 0]
    # 7 -> [0, 0, 0, 0, 1, 0, 0]
    y_one_hot[i, int(label) - 3] = 1

outputs_num = y_one_hot.shape[1]
inputs_num = D

X_train, X_test, y_train, y_test = train_test_split(
    X, y_one_hot, test_size=config["data"]["test_size"], random_state=SEED
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit + transform for train data
X_test = scaler.transform(X_test)  # Only transform for test data


# Implemented neural network


class NeuralNetworkWineQuality:
    def __init__(self, X, y, layers):
        self.X = X
        self.y = y
        self.layers = layers
        self.weights = []
        self.biases = []

        # Initiation of weights and biases for each layer
        for i in range(len(layers) - 1):
            limit = np.sqrt(1 / layers[i])
            self.weights.append(
                np.random.uniform(-limit, limit, (layers[i], layers[i + 1]))
            )
            self.biases.append(np.zeros((1, layers[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        """Forward pass in the network"""
        self.activations = [X]
        self.z_values = []

        for i in range(len(self.weights)):
            neurons = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(neurons)
            if i < len(self.weights) - 1:
                a = self.sigmoid(neurons)
            else:
                a = self.softmax(neurons)  # Softmax fot output layer
            self.activations.append(a)

        return self.activations[-1]

    def backward(self, X, y, y_pred):
        """Backward pass in the network"""
        m = X.shape[0]
        dz = y_pred - y

        for i in range(len(self.weights) - 1, -1, -1):
            dw = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m

            if i > 0:
                dz = np.dot(dz, self.weights[i].T) * self.sigmoid_derivative(
                    self.activations[i]
                )

            # Update weights and biases
            self.weights[i] -= self.learning_rate * dw
            self.biases[i] -= self.learning_rate * db

    def train(self, epochs, batch_size, learning_rate):
        """Algorithm for training the network with backpropagation and mini-batch"""
        self.learning_rate = learning_rate
        self.best_weights = None  # Store best weights and biases
        self.best_biases = None
        self.loss = self.compute_loss(self.y, self.forward(self.X))

        for epoch in range(epochs):
            # Random permutation of the data
            permutation = np.random.permutation(self.X.shape[0])
            X_shuffled = self.X[permutation]
            y_shuffled = self.y[permutation]

            for i in range(0, self.X.shape[0], batch_size):
                X_batch = X_shuffled[i: i + batch_size]
                y_batch = y_shuffled[i: i + batch_size]

                # Forward pass
                y_pred = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, y_pred)

            # Optional: print loss every 10 epochs
            loss = self.compute_loss(self.y, self.forward(self.X))
            if loss < self.loss:
                self.loss = loss
                self.best_weights = [w.copy() for w in self.weights]
                self.best_biases = [b.copy() for b in self.biases]
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, loss: {loss:.4f}")

        self.weights = self.best_weights
        self.biases = self.best_biases

    def compute_loss(self, y, y_pred):
        """Cross-entropy loss function"""
        return -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))

    def classify(self, y):
        """Classify the output of the network"""
        return np.argmax(y, axis=1).reshape(-1, 1) + 3

    def accuracy(self, X, y):
        y_pred = self.forward(X)
        y_pred_class = self.classify(y_pred)
        y_true_class = self.classify(y)
        return np.mean(y_pred_class == y_true_class)

    def get_best_weights_and_biases(self):
        return self.best_weights, self.best_biases

    def show_predicitons(self, X, y):
        y_pred = self.forward(X)
        y_pred_class = self.classify(y_pred)
        y_true_class = self.classify(y)
        return y_pred_class, y_true_class
    
    def predict(self, X):
        y_pred = self.forward(X)
        y_pred_class = self.classify(y_pred)
        return y_pred_class


# Network parameters
layers = [inputs_num] + config["hidden_layers"]+  [outputs_num]  # Layers: input -> hidden1 -> ... -> output
nn = NeuralNetworkWineQuality(X_train, y_train, layers)

# Training
nn.train(epochs=config["training"]["epochs"], batch_size=config["training"]["batch_size"], learning_rate=config["training"]["learning_rate"])

# Testing
accuracy = nn.accuracy(X_train, y_train)
print(f"Accuracy on train set: {accuracy:.4f}")
accuracy = nn.accuracy(X_test, y_test)
print(f"Accuracy on test set: {accuracy:.4f}")
y_pred = nn.predict(X_test)
y_true_class = nn.classify(y_test)
mae = mean_absolute_error(y_true_class, y_pred)
print(f"Mean absolute error: {mae:.2f}")
# Predictions
print("Predictions on test set:")
y_pred_class, y_true_class = nn.show_predicitons(X_test, y_test)
df_test = pd.DataFrame(
    np.concatenate((y_pred_class, y_true_class), axis=1), columns=["Predicted", "True"]
)
print(df_test)

from random import uniform


def classify(row, weights):
    """
    Compute dot product of row features with weights and return 0 or 1.
    weights[-1] is the bias term; row[-1] is the label (ignored here).
    """
    activation = weights[-1]  # bias
    for i in range(len(row) - 1):
        activation += weights[i] * row[i]
    return 1 if activation >= 0.0 else 0


def train(train_data, n_epoch, l_rate=1):
    """
    Train a binary perceptron using stochastic gradient descent.
    train_data: list of rows where row[-1] is the label (0 or 1).
    Returns the trained weight vector.
    """
    # Initialize weights (including bias at index -1) with small random values
    weights = [uniform(-1, 1) for _ in range(len(train_data[0]))]

    for n in range(n_epoch):
        accuracy = 0
        for i in range(len(train_data)):
            c = classify(train_data[i], weights)
            if train_data[i][-1] != c:
                error = train_data[i][-1] - c
                # Update each feature weight
                for p in range(len(weights) - 1):
                    weights[p] += l_rate * error * train_data[i][p]
                # Update bias ONCE per sample, outside the feature loop
                weights[-1] += l_rate * error
            else:
                accuracy += 1 / len(train_data)
        print("epoch", n, "...", round(accuracy * 100, 4), "%")

    return weights

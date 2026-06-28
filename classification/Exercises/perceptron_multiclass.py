import Helpers.util
from random import uniform

class PerceptronClassifier:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = None

    def classify(self, row):
        best_label = None
        best_score = float('-inf')
        for idx, label in enumerate(self.legalLabels):
            score = 0
            for i in range(len(row)):
                score += row[i] * self.weights[idx][i]
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def train(self, train_data, labels):
        n_classes = len(self.legalLabels)
        n_features = len(train_data[0])
        self.weights = [
            [uniform(-1, 1) for _ in range(n_features)]
            for _ in range(n_classes)
        ]
        for n in range(self.epochs):
            accuracy = 0
            for i in range(len(train_data)):
                pred = self.classify(train_data[i])
                actual = labels[i]
                pred_idx   = self.legalLabels.index(pred)
                actual_idx = self.legalLabels.index(actual)
                if pred != actual:
                    for j in range(n_features):
                        self.weights[pred_idx][j]   -= train_data[i][j]
                        self.weights[actual_idx][j] += train_data[i][j]
                else:
                    accuracy += 1 / len(train_data)
            print("epoch", n, "...", round(accuracy * 100, 4), "%")

    def findHighWeightFeatures(self, label):
        idx = self.legalLabels.index(label)
        weight_pairs = list(enumerate(self.weights[idx]))
        weight_pairs.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in weight_pairs[:100]]
    
    
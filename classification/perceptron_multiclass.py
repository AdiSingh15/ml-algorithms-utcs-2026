import Helpers.util
from random import uniform


class PerceptronClassifier:

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.epochs = max_iterations
        self.weights = None  # initialized in train()

    def classify(self, row):
        """
        Return the label with the highest dot-product score.
        Called externally by the Berkeley harness with just the data point.
        """
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
        """
        Train one weight vector per class using the multiclass perceptron rule.
        On a mistake: penalize the predicted class AND reward the actual class.
        Both updates happen inside the same `if pred != actual` block.
        """
        n_classes = len(self.legalLabels)
        n_features = len(train_data[0])

        # One weight vector per class, randomly initialized
        self.weights = [
            [uniform(-1, 1) for _ in range(n_features)]
            for _ in range(n_classes)
        ]

        for n in range(self.epochs):
            accuracy = 0
            for i in range(len(train_data)):
                pred = self.classify(train_data[i])
                actual = labels[i]

                if pred != actual:
                    pred_idx   = self.legalLabels.index(pred)
                    actual_idx = self.legalLabels.index(actual)
                    for j in range(n_features):
                        # Penalize the wrongly predicted class
                        self.weights[pred_idx][j]   -= train_data[i][j]
                        # Reward the correct class -- both updates in the if block
                        self.weights[actual_idx][j] += train_data[i][j]
                else:
                    accuracy += 1 / len(train_data)

            print("epoch", n, "...", round(accuracy * 100, 4), "%")

    def findHighWeightFeatures(self, label):
        """
        Return the 100 features with the highest weight for the given label.
        Used by the Berkeley harness for visualization.
        """
        idx = self.legalLabels.index(label)
        weight_pairs = list(enumerate(self.weights[idx]))
        weight_pairs.sort(key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in weight_pairs[:100]]
      

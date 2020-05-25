from collections import Counter

class Accuracy:
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual
    def accuracy(self):
        return (self.predicted == self.actual).mean()
    def F_score(self):
        counter = Counter(zip(self.predicted, self.actual))
        self.truePositive = counter[1, 1]
        self.falsePositive = counter[1, 0]
        self.trueNegative = counter[0, 0]
        self.falseNegative = counter[0, 1]
        
        self.precision = self.truePositive / (self.truePositive + self.falsePositive)
        self.recall = self.truePositive / (self.truePositive + self.falseNegative)
        self.fScore = (2 * self.precision * self.recall) / (self.precision + self.recall)
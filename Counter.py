from DecisionTreeManager import DecisionTreeManager

class Counter:

    def __init__(self, exdtm: DecisionTreeManager):
        self.x = None
        self.y = None
        self.dtm = exdtm

    def setx(self, x):
        self.x = x

    def sety(self, y):
        self.y = y

    def count_accuracy(self):
        acc = 0
        for i in range(0, self.x.shape[0]):
            result = self.dtm.predict_class(self.x[i])
            if result == self.y[i]:
                acc += 1
        print(f"Accuracy: {(acc / (self.y.shape[0])) * 100}%")
        print()
import NodeOfTree
import traceback

class DecisionTreeManager:

    def __init__(self):
        self.root = None

    def train(self, x, y):
        try:
            self.root = NodeOfTree.NodeOfTree()
            self.root.train(x, y, 0)
        except Exception:
            print("Something happened while training Nodes")
            print(traceback.format_exc())

    def predict_class(self, X):
        try:
            result = self.root.classify(X)
            return result
        except Exception:
            print("Decision tree manager has raised a class prediction exception")
            print(traceback.format_exc())

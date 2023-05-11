import DecisionTreeManager
import LoadData
import Counter

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = LoadData.load_data()
    DTManager = DecisionTreeManager.DecisionTreeManager()
    DTManager.train(x_train, y_train)

    counter = Counter.Counter(DTManager)
    counter.setx(x_train)
    counter.sety(y_train)
    print("Train:")
    counter.count_accuracy()
    # True positive + true negative / all objects


    counter.setx(x_test)
    counter.sety(y_test)
    print("Test: ")
    counter.count_accuracy()
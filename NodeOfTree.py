import numpy
import numpy as np
import traceback

TREEDEPTH = 5


class NodeOfTree:

    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.split_column_index = None
        self.split_value = None
        self.prediction_value = None
        self.if_leaf = False

    def calculate_gini_subset(self, sum1, sum2, sum_total):
        end_sum1 = sum1 / sum_total
        end_sum1 **= 2
        end_sum2 = sum2 / sum_total
        end_sum2 **= 2
        result = 1 - (end_sum1 + end_sum2)
        return result

    def calculate_gini_total(self, sum1, sum2, gini1, gini2, sum_total):
        prob_gini1 = (sum1 / sum_total) * gini1
        prob_gini2 = (sum2 / sum_total) * gini2

        result = prob_gini1 + prob_gini2
        return result

    def least_gini_value(self, possible_splits, sorted_y):
        least_gini = np.inf
        least_gini_index = None

        # for each possible split
        for split in possible_splits:
            # calculate number of recommended and notrecommended tracks for each subset of split
            # conditions in sum are as follows:
            # if recommended/notrecommended and if in first/second subset

            notrecommended1 = sum([1 for j in range(np.shape(sorted_y)[0]) if sorted_y[j, 0] == 0 and
                                   split >= j])
            recommended1 = sum([1 for j in range(np.shape(sorted_y)[0]) if sorted_y[j, 0] == 1 and
                                split >= j])
            notrecommended2 = sum([1 for j in range(np.shape(sorted_y)[0]) if sorted_y[j, 0] == 0 and
                                   split < j])
            recommended2 = sum([1 for j in range(np.shape(sorted_y)[0]) if sorted_y[j, 0] == 1 and
                                split < j])


            sum_subset1 = notrecommended1 + recommended1
            sum_subset2 = notrecommended2 + recommended2

            # calculate gini for subsets
            gini_subset1 = self.calculate_gini_subset(recommended1, notrecommended1, sum_subset1)
            gini_subset2 = self.calculate_gini_subset(recommended2, notrecommended2, sum_subset2)

            gini_total = self.calculate_gini_total(sum_subset1, sum_subset2, gini_subset1, gini_subset2, sum_subset1 +
                                                   sum_subset2)

            if least_gini > gini_total:
                least_gini = gini_total
                least_gini_index = split

        return least_gini, least_gini_index

    def best_split(self, x, y):
        best_gain = numpy.inf
        index_of_best_gain = None
        split_value = None

        # for each column in x array
        for column_index in range(numpy.shape(x)[1]):
            order = numpy.argsort(x[:, column_index])
            sorted_x = x[order]
            sorted_y = y[order]

            possible_splits = []
            # for each row except for the last
            for row_index in range(numpy.shape(sorted_x)[0] - 1):
                if sorted_x[row_index, column_index] != sorted_x[row_index + 1, column_index]:
                    possible_splits.append(row_index)

                gini_best_value, gini_best_index = self.least_gini_value(possible_splits, sorted_y)

            if best_gain > gini_best_value:
                best_gain = gini_best_value
                index_of_best_gain = column_index
                split_value = np.mean(x[[gini_best_index, gini_best_index+1], column_index])

        if index_of_best_gain is None:
            raise Exception

        if split_value is None:
            raise Exception

        return split_value, index_of_best_gain

    def split(self, x, y):
        sort = np.argsort(x[:, self.split_column_index])

        x_sorted = x[sort]
        y_sorted = y[sort]

        condition_for_left = x_sorted[:, self.split_column_index] < self.split_value
        condition_for_right = x_sorted[:, self.split_column_index] >= self.split_value

        x_left = x_sorted[condition_for_left]
        y_left = y_sorted[condition_for_left]

        x_right = x_sorted[condition_for_right]
        y_right = y_sorted[condition_for_right]

        return x_left, y_left, x_right, y_right

    # recursive function for creation of the tree
    def train(self, x, y, depth):

        self.prediction_value = numpy.sum(y) / y.size
        # if depth is at a maximal level or all members of node are one class
        if self.prediction_value == 0.0 or self.prediction_value == 1.0 or depth == TREEDEPTH:
            self.if_leaf = True
            return True

        # find the best split
        try:
            self.split_value, self.split_column_index = self.best_split(x, y)
        except Exception:
            print("Best split function raised an Exception due to unhandled case")
            raise Exception

        # split data into subsets
        x_left, y_left, x_right, y_right = self.split(x, y)
        if x_left.size == 0 or y_left.size == 0 or x_right.size == 0 or y_right.size == 0:
            self.if_leaf = True
            return True

        new_depth = depth + 1
        self.left_child, self.right_child = NodeOfTree(), NodeOfTree()
        self.left_child.train(x_left, y_left, new_depth)
        self.right_child.train(x_right, y_right, new_depth)

    # classifying the element by the x
    def classify(self, x):
        try:
            if self.if_leaf is True:
                return round(self.prediction_value)
            elif x[self.split_column_index] <= self.split_value:
                return self.left_child.classify(x)
            elif x[self.split_column_index] > self.split_value:
                return self.right_child.classify(x)
            else:  # if something goes wrong
                raise Exception
        except Exception:
            print("Decision tree manager has raised a class prediction exception")
            print(traceback.format_exc())
            print(self.split_column_index)
            print(self.split_value)
            print(self.right_child)
            print(self.left_child)
            print()

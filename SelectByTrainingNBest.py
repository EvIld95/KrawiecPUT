import itertools

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


class SelectByTrainingNBest:
    def __init__(self, n_best, X, y):
        self.n_best = n_best
        self.X = X
        self.y = y

    def transform(self):
        number_of_attr = len(self.X[0])
        attr_list = list(range(0, number_of_attr))
        best_attr_list = []
        for best_attr_number in range(1, self.n_best + 1):
            attr_combinations = [list(l) for l in itertools.combinations(attr_list, best_attr_number)]
            score_list = dict()
            for combination in attr_combinations:
                if self.check_combination_contains_best_attr(combination, best_attr_list):
                    X_train = self.X[:, combination]
                    score_list[tuple(combination)] = (self.count_result(X_train, self.y))
            best_attr_list = max(score_list, key=score_list.get)
        return self.X[:, best_attr_list]

    def count_result(self, X_train, y_train):
        classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
        scores = cross_val_score(classifier, X_train, y_train, cv=2)
        result = scores.mean()
        # print("Result: ", result)
        return result

    def check_combination_contains_best_attr(self, combination, best_attr_list):
        if best_attr_list:
            return all(x in combination for x in best_attr_list)
        else:
            return True

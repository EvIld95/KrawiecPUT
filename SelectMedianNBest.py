import numpy as np
from collections import OrderedDict


class SelectMedianNBest:
    def __init__(self, n_best, X, y):
        self.n_best = n_best
        self.X = X
        self.y = y

    def transform(self):
        letter_set = dict()
        letter_mean_set = dict()
        groups_per_attr = dict()
        sorted_letter_count_set = dict()
        number_of_attr = len(self.X[0])
        for attr_index in range(0, number_of_attr):
            attr = self.X[:, attr_index]
            for letter_id, letter in enumerate(self.y):
                if letter not in letter_set.keys():
                    letter_set[letter] = [attr[letter_id]]
                else:
                    letter_set[letter].append(attr[letter_id])
            for key, values in letter_set.items():
                letter_mean_set[key] = np.median(letter_set[key])
            sorted_letter_set = OrderedDict(sorted(letter_mean_set.items(), key=lambda t: t[1]))
            for key, value in sorted_letter_set.items():
                if value not in sorted_letter_count_set:
                    sorted_letter_count_set[value] = 1
                else:
                    sorted_letter_count_set[value] += 1
            groups_per_attr[attr_index] = len(sorted_letter_count_set)
        groups_per_attr = OrderedDict(sorted(groups_per_attr.items(), key=lambda t: t[1], reverse=True))
        n_best_attr_groups_with_values = {k: groups_per_attr[k] for k in list(groups_per_attr)[:self.n_best]}
        n_best_attr_groups = list(n_best_attr_groups_with_values.keys())
        return self.X[:, n_best_attr_groups]

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0].values
y = y[:1000]

letter_set = dict()
letter_mean_set = dict()

number_of_subplots = 8
plt.figure(figsize=(8, 10))

for attr_index in range(1, number_of_subplots+1):
    attr = X[:1000, attr_index]
    for letter_id, letter in enumerate(y):
        if letter not in letter_set.keys():
            letter_set[letter] = [attr[letter_id]]
        else:
            letter_set[letter].append(attr[letter_id])

    # print(letter_set)

    for key, values in letter_set.items():
        letter_mean_set[key] = np.median(letter_set[key])

    sorted_letter_set = OrderedDict(sorted(letter_mean_set.items(), key=lambda t: t[1]))

    plt.subplot(4, 2, attr_index)

    for key, value in sorted_letter_set.items():
        plt.scatter(key, value, marker='o', s=5, c='b')

# plt.tight_layout()
plt.show()
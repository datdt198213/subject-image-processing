"""
To calculate probabilities in an array
To support histogram equalization
"""

import numpy as np


# Add array to dictionary
def add_dict(array):
    probabilities = {}
    for item in array:
        for i in item:
            if i not in probabilities:
                probabilities.update({i: 1})
            else:
                count = probabilities.get(i)
                count += 1
                probabilities[i] = count
    return probabilities


def sort_dict(dictionary):
    my_keys = list(dictionary.keys())
    my_keys.sort()
    sorted_dict = {i: dictionary[i] for i in my_keys}
    return sorted_dict


# Compute probabilities
def compute(array):
    size = array.size
    dictionary = add_dict(array)

    for key in dictionary.keys():
        value = dictionary[key]
        value = value / size
        dictionary[key] = value

    sorted_dict = sort_dict(dictionary)
    print(sorted_dict)


def main():
    array = np.array([[6, 6, 7, 7, 2, 1, 1, 1],
                      [7, 7, 6, 6, 2, 1, 1, 1],
                      [7, 7, 6, 6, 1, 1, 2, 0],
                      [7, 7, 6, 6, 7, 7, 8, 8],
                      [7, 7, 7, 7, 8, 9, 8, 7],
                      [7, 6, 7, 7, 7, 8, 9, 7],
                      [7, 7, 7, 7, 1, 1, 1, 2],
                      [6, 6, 6, 7, 1, 1, 0, 0]])
    compute(array)


if __name__ == "__main__":
    main()

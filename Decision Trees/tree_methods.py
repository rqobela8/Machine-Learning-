import math
from typing import List, Any
from collections import Counter

def entropy(class_probs:List[float])->float:
     """Given a list of class probabilities, compute the entropy"""

     return sum(-p*math.log(p,2)
                for p in class_probs
                if p>0)

assert entropy([1.0]) == 0
assert entropy([0.5, 0.5]) == 1
assert 0.81 < entropy([0.25, 0.75]) < 0.82

def class_probabilities(data:List[Any])->List[float]:
    """
    Calculate fraction of each class in dataset

    :param data: Dataset
    :return: list of probabilities
    """

    total_examples = len(data)

    return [count/total_examples
            for count in Counter(data).values()] #Counter(data): Dict[Any,int]


def data_entropy(labels:List[Any])->float:
    """
    Returns entropy of dataset

    :param labels: Dataset
    :return: entropy
    """

    return entropy(class_probabilities(labels))


assert data_entropy(['a']) == 0
assert data_entropy([True, False]) == 1
assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])


def partition_entropy(subsets: List[List[Any]]) -> float:
    """Returns the entropy from this partition of data into subsets"""
    total_count = sum(len(subset) for subset in subsets)
    return sum(data_entropy(subset) * len(subset) / total_count
    for subset in subsets)


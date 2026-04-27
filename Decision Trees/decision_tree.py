from collections import defaultdict
from typing import NamedTuple, Optional, TypeVar, List, Dict, Any, Counter
from tree_methods import partition_entropy


class Candidate(NamedTuple):
    level: str
    lang: str
    tweets: bool
    phd: bool
    did_well: Optional[bool] = None  # allow unlabeled data


#Data
                   # level     lang   tweets  phd   did_well
inputs = [Candidate('Senior', 'Java', False, False, False),
Candidate('Senior', 'Java', False, True, False),
Candidate('Mid', 'Python', False, False, True),
Candidate('Junior', 'Python', False, False, True),
Candidate('Junior', 'R', True, False, True),
Candidate('Junior', 'R', True, True, False),
Candidate('Mid', 'R', True, True, True),
Candidate('Senior', 'Python', False, False, False),
Candidate('Senior', 'R', True, False, True),
Candidate('Junior', 'Python', True, False, True),
Candidate('Senior', 'Python', True, True, True),
Candidate('Mid', 'Python', False, True, True),
Candidate('Mid', 'Java', True, False, True),
Candidate('Junior', 'Python', False, True, False)
]

T = TypeVar("T") # generic type for inputs

def partition_by(data:List[T],attribute:str)->Dict[Any,List[T]]:
    """
    Partition the inputs into lists based on the specified attribute.

    :param attribute: feature to partition
    :param data: Any type of data
    :return: Dict where key is feature and value is list of data objects.
    """

    partitions:Dict[Any,List[T]] = defaultdict(list)

    for example in data:
        key = getattr(example,attribute)
        partitions[key].append(example)
    return partitions


def partition_entropy_by(inputs:List[T],
                         attribute:str,
                         label_attribute:str)->float:
    """Compute the entropy corresponding to the given partition"""
    # partitions consist of our inputs

    partitions = partition_by(inputs,attribute)

    labels = [[getattr(example,label_attribute) for example in partition]
              for partition in partitions.values()]

    return partition_entropy(labels)




if __name__ == "__main__":
    assert 0.69 < partition_entropy_by(inputs, 'level', 'did_well') < 0.70
    assert 0.86 < partition_entropy_by(inputs, 'lang', 'did_well') < 0.87
    assert 0.78 < partition_entropy_by(inputs, 'tweets', 'did_well') < 0.79
    assert 0.89 < partition_entropy_by(inputs, 'phd', 'did_well') < 0.90

    senior_inputs = [input for input in inputs if input.level == 'Senior']

    for key in ['lang', 'tweets', 'phd']:
        print(key, partition_entropy_by(senior_inputs, key, 'did_well'))

    assert 0.4 == partition_entropy_by(senior_inputs, 'lang', 'did_well')
    assert 0.0 == partition_entropy_by(senior_inputs, 'tweets', 'did_well')
    assert 0.95 < partition_entropy_by(senior_inputs, 'phd', 'did_well') < 0.96

    senior_tweet_inputs = [
        inp for inp in inputs
        if inp.level == 'Senior' and inp.tweets == True
    ]

    for key in ['lang', 'phd']:
        print(key, partition_entropy_by(senior_tweet_inputs, key, 'did_well'))

    label_counts = Counter(getattr(input, "did_well")
                           for input in inputs)
    print(label_counts)

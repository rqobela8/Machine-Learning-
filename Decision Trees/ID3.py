from typing import NamedTuple, Any, List, Dict, Counter
from decision_tree import partition_entropy_by,partition_by,Candidate

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


class Split(NamedTuple):
    attribute:str
    subtrees:dict
    default_value:Any = None


class Leaf(NamedTuple):
    value:Any


def build_tree_id3(data:List[Any],
                   split_attributes:List[str],
                   target_attribute:str):
    # Count target labels
    label_counts = Counter(getattr(example, target_attribute)
                           for example in data)
    most_common_label = label_counts.most_common(1)[0][0]

    # If there's a unique label, predict it
    if len(label_counts) == 1:
        return Leaf(most_common_label)

    # If no split attributes left, return the majority label
    if not split_attributes:
        return Leaf(most_common_label)

    # Otherwise split by the best attribute
    def split_entropy(attribute: str) -> float:
        """Helper function for finding the best attribute"""
        return partition_entropy_by(data, attribute, target_attribute)

    best_attribute = min(split_attributes, key=split_entropy)

    partitions = partition_by(data, best_attribute)
    new_attributes = [a for a in split_attributes if a != best_attribute]

    # Recursively build the subtrees
    subtrees = {attribute_value: build_tree_id3(subset,
                                                new_attributes,
                                                target_attribute)
                for attribute_value, subset in partitions.items()}
    print(subtrees)

    return Split(best_attribute, subtrees, default_value=most_common_label)

def classify(tree: Any, input: Any) -> Any:
    """classify the input using the given decision tree"""
    # If this is a leaf node, return its value
    if isinstance(tree, Leaf):
        return tree.value
    # Otherwise this tree consists of an attribute to split on
    # and a dictionary whose keys are values of that attribute
    # and whose values are subtrees to consider next

    subtree_key = getattr(input, tree.attribute)
    if subtree_key not in tree.subtrees:  # If no subtree for key,
        return tree.default_value  # return the default value.

    subtree = tree.subtrees[subtree_key]  # Choose the appropriate subtree
    return classify(subtree, input)  # and use it to classify the input.


if __name__ == "__main__":
  tree = build_tree_id3(inputs,
['level', 'lang', 'tweets', 'phd'],
'did_well')
  # Should predict True
  assert classify(tree, Candidate("Junior", "Java", True, False))
  # Should predict False
  assert not classify(tree, Candidate("Junior", "Java", True, True))










class Node:
    def __init__(self, layer, index):
        self.layer = layer
        self.index = index
        self.loc = (layer, index)


class Interval:
    def __init__(self, lb, ub):
        self.lb = lb
        self.ub = ub


class Inn:
    """
    args:
    weights: dict of {(Node, Node), Interval}
    biases: dict of {Node, Interval}
    """
    def __init__(self, num_layers, delta):
        self.num_layers = num_layers
        self.delta = delta
        self.weights = dict()
        self.biases = dict()

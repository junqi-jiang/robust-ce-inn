

class Interval:
    def __init__(self, lb=0, ub=0):
        self.lb = lb
        self.ub = ub

    def set_bounds(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def get_bound(self, y_prime):
        return self.lb if y_prime is 1 else self.ub


class Node(Interval):
    def __init__(self, layer, index, lb=0, ub=0):
        super().__init__(lb, ub)
        self.layer = layer
        self.index = index
        self.loc = (layer, index)


class Inn:
    """
    args:
    nodes: dict of {int layer num, [Node1, Node2, ...]}
    weights: dict of {(Node in prev layer, Node in this layer), Interval}.
    biases: dict of {Node, Interval}
    """
    def __init__(self, num_layers, delta):
        self.num_layers = num_layers
        self.delta = delta
        self.nodes = dict()
        self.weights = dict()
        self.biases = dict()

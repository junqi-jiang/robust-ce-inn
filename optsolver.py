"""
Author: Jay Jiang
Date: 20 June 2022
"""

from gurobipy import *
from dataset import *
from inn import *


class OptSolver:
    def __init__(self, dataset, inn, y_prime):
        self.dataset = dataset
        self.inn = inn
        self.y_prime = y_prime
        self.model = Model()  # initialise Gurobi optimisation model
        self.node_vars = dict()     # dict of {layer number, {Node, Gurobi variable obj}}

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx in range(self.dataset.num_features):
            # cases by feature type, add different constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                disc_var_list = []
                for var_idx in self.dataset.feat_var_map[feat_idx]:
                    node_var[var_idx] = self.model.addVar(vtype = GRB.BINARY, name='x_0_' + str(var_idx))
                    disc_var_list.append(node_var[var_idx])
                self.model.addConstr(quicksum(disc_var_list) == 1, name="?")

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    node_var[var_idx] = self.model.addVar(vtype = GRB.BINARY, name='x_0_' + str(var_idx))
                    if i == 0:
                        prev_var = node_var[var_idx]
                        continue
                    self.model.addConstr(prev_var >= node_var[var_idx])

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                node_var[var_idx] = self.model.addVar(lb=0, ub=1, vtype=GRB.SEMICONT)

        self.node_vars[0] = node_var

    def generate_node_variables(self):
        # create variables
        pass

    def add_hidden_constraints(self):
        pass

    def add_output_constraints(self):
        pass

    def compute(self):
        pass

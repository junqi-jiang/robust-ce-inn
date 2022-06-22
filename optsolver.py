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
        self.y_prime = y_prime  # if 0, constraint: upper output node < 0.5, if 1, constraint: lower output node >= 0.5
        self.model = Model()  # initialise Gurobi optimisation model

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx in range(self.dataset.num_features):
            # cases by feature type, add different types of variables and constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                disc_var_list = []
                for var_idx in self.dataset.feat_var_map[feat_idx]:
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_0_' + str(var_idx))
                    disc_var_list.append(node_var[var_idx])
                self.model.update()
                self.model.addConstr(quicksum(disc_var_list) == 1)

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_0_' + str(var_idx))
                    if i == 0:
                        prev_var = node_var[var_idx]
                        continue
                    self.model.update()
                    self.model.addConstr(prev_var >= node_var[var_idx])

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                node_var[var_idx] = self.model.addVar(lb=0, ub=1, vtype=GRB.SEMICONT)
                self.model.update()
            self.model.update()
        return node_var

    def add_node_variables_constraints(self, node_vars, aux_vars):
        """
        create variables for nodes. Each node has the followings:
        node variable n for the final node value after ReLU,
        affine variable a for the node value before ReLU.

        Constraint on each node:
        a = WX+B
        n = max{a, 0}
        """
        for i in range(1, self.inn.num_layers):
            node_var = dict()
            aux_var = dict()
            for node in self.inn.nodes[i]:
                node_var[node.index] = self.model.addVar(vtype=GRB.SEMICONT, name='x_' + str(i) + '_' + str(node.index))
                self.model.update()
                # add auxiliary variable for hidden layer nodes
                if i != (self.inn.num_layers - 1):
                    aux_var[node.index] = self.model.addVar(vtype=GRB.SEMICONT,
                                                            name='a_' + str(i) + '_' + str(node.index))
                    self.model.update()
                    self.model.addConstr(aux_var[node.index] == quicksum(
                        self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index] for
                        node1 in self.inn.nodes[i - 1]) + node.get_bound(self.y_prime) == aux_var[node.index])
                    self.model.addConstr(node_var[node.index] == max_(0, aux_var[node.index]))

                # add output constraint
                else:
                    self.model.addConstr(aux_var[node.index] == quicksum(
                        self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index] for
                        node1 in self.inn.nodes[i - 1]) + node.get_bound(self.y_prime) == node_var[node.index])
                    if self.y_prime:
                        self.model.addConstr(node_var[node.index] >= 0.5)
                    else:
                        self.model.addConstr(node_var[node.index] > 0.5)
                self.model.update()
            node_vars[i] = node_var
            if i != (self.inn.num_layers - 1):
                aux_vars[i] = aux_var
        return node_vars, aux_vars

    def create_constraints(self):
        node_vars = dict()  # dict of {layer number, {Node idx, Gurobi variable obj}}
        aux_vars = dict()  # dict of {layer number, {Node idx, Gurobi variable obj}}
        node_vars[0] = self.add_input_variable_constraints()
        node_vars, aux_vars = self.add_node_variables_constraints(node_vars, aux_vars)

    def compute(self):
        pass

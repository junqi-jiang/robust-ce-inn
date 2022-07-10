"""
Author: Junqi Jay Jiang
Date: 20 June 2022
"""

from gurobipy import *
from dataset import *
from inn import *


class OptSolver:
    def __init__(self, dataset, inn, y_prime, x):
        self.dataset = dataset
        self.inn = inn
        self.y_prime = y_prime  # if 0, constraint: upper output node < 0, if 1, constraint: lower output node >= 0
        self.x = x  # explainee instance x
        self.model = Model()  # initialise Gurobi optimisation model

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx in range(self.dataset.num_features):
            # cases by feature type, add different types of variables and constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                disc_var_list = []
                for var_idx in self.dataset.feat_var_map[feat_idx]:
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_disc_0_' + str(var_idx))
                    disc_var_list.append(node_var[var_idx])
                self.model.update()
                self.model.addConstr(quicksum(disc_var_list) == 1)

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_ord_0_' + str(var_idx))
                    if i == 0:
                        prev_var = node_var[var_idx]
                        continue
                    self.model.update()
                    self.model.addConstr(prev_var >= node_var[var_idx])

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                node_var[var_idx] = self.model.addVar(lb=0, ub=1, vtype=GRB.SEMICONT, name="x_cont_0" + str(var_idx))
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
        eps = 0.0001
        for i in range(1, self.inn.num_layers):
            node_var = dict()
            aux_var = dict()
            for node in self.inn.nodes[i]:
                node_var[node.index] = self.model.addVar(vtype=GRB.CONTINUOUS, name='n_' + str(node))
                self.model.update()
                # add auxiliary variable for hidden layer nodes
                if i != (self.inn.num_layers - 1):
                    aux_var[node.index] = self.model.addVar(vtype=GRB.CONTINUOUS,
                                                            name='a_' + str(node))
                    self.model.update()
                    # node1 are nodes in prev layer
                    self.model.addConstr(aux_var[node.index] == quicksum(
                        self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index] for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].get_bound(self.y_prime),
                                         name="forward_pass_node_" + str(node))
                    self.model.addConstr(node_var[node.index] == max_(0, aux_var[node.index]),
                                         name="after_relu_node_" + str(node))

                # add output constraint
                else:
                    self.model.addConstr(node_var[node.index] == quicksum(
                        self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index] for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].get_bound(self.y_prime),
                                         name="forward_pass_output_node_" + str(node))
                    if self.y_prime:
                        self.model.addConstr(node_var[node.index] - eps >= 0.0, name="output_node_lb_>=0")
                    else:
                        self.model.addConstr(node_var[node.index] <= 0.0 - eps, name="output_node_ub_<0")
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
        return node_vars, aux_vars

    def compute(self):
        node_vars, aux_vars = self.create_constraints()
        inputs = []
        for key in node_vars[0].keys():
            inputs.append(node_vars[0][key])
        self.model.setObjective(quicksum(inputs-self.x), GRB.MINIMIZE)
        self.model.update()
        self.model.optimize()
        for v in self.model.getVars():
            print(v.varName, v.getAttr(GRB.Attr.X))


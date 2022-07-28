"""
Author: Junqi Jay Jiang
Date: 20 June 2022
"""

from gurobipy import *
from dataset import *
import numpy as np


class OptSolver:
    def __init__(self, dataset, inn, y_prime, x, eps=0.0001):
        self.dataset = dataset
        self.inn = inn
        self.y_prime = y_prime  # if 0, constraint: upper output node < 0, if 1, constraint: lower output node >= 0
        self.x = x  # explainee instance x
        self.model = Model()  # initialise Gurobi optimisation model
        self.x_prime = None  # counterfactual instance
        self.eps = eps

    def add_input_variable_constraints(self):
        node_var = dict()
        for feat_idx in range(self.dataset.num_features):
            # cases by feature type, add different types of variables and constraints
            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                disc_var_list = []
                for var_idx in self.dataset.feat_var_map[feat_idx]:
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_disc_0_' + str(var_idx), ub=1, lb=0)
                    disc_var_list.append(node_var[var_idx])
                self.model.update()
                self.model.addConstr(quicksum(disc_var_list) == 1, name='x_disc_0_feat' + str(feat_idx))

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                prev_var = None
                ord_var_list = []
                for i, var_idx in enumerate(self.dataset.feat_var_map[feat_idx]):
                    node_var[var_idx] = self.model.addVar(vtype=GRB.BINARY, name='x_ord_0_' + str(var_idx), ub=1, lb=0)
                    self.model.update()
                    if i != 0:
                        self.model.addConstr(prev_var >= node_var[var_idx],
                                             name='x_ord_0_var' + str(var_idx - 1) + '_geq_' + str(var_idx))
                    prev_var = node_var[var_idx]
                    ord_var_list.append(node_var[var_idx])
                self.model.addConstr(quicksum(ord_var_list) >= 1, name='x_ord_0_feat' + str(feat_idx) + '_geq1')

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                var_idx = self.dataset.feat_var_map[feat_idx][0]
                node_var[var_idx] = self.model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name="x_cont_0_" + str(var_idx))

            self.model.update()
        return node_var

    def add_node_variables_constraints(self, node_vars, aux_vars):
        """
        create variables for nodes. Each node has the followings:
        node variable n for the final node value after ReLU,
        auxiliary variable a for the node value before ReLU.

        Constraint on each node:
        a = WX+B
        n = max{a, 0}
        """
        for i in range(1, self.inn.num_layers):
            node_var = dict()
            aux_var = dict()
            for node in self.inn.nodes[i]:

                self.model.update()
                # add auxiliary variable for hidden layer nodes
                if i != (self.inn.num_layers - 1):
                    node_var[node.index] = self.model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='n_' + str(node))
                    aux_var[node.index] = self.model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                            name='a_' + str(node))
                    self.model.update()
                    # node1 are nodes in prev layer
                    self.model.addConstr(aux_var[node.index] == quicksum(
                        (self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].get_bound(self.y_prime),
                                         name="forward_pass_node_" + str(node))
                    self.model.addConstr(node_var[node.index] == max_(0, aux_var[node.index]),
                                         name="after_relu_node_" + str(node))

                # add output constraint: y = WX+b,
                else:
                    node_var[node.index] = self.model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                                             name='n_' + str(node))
                    self.model.addConstr(node_var[node.index] == quicksum(
                        (self.inn.weights[(node1, node)].get_bound(self.y_prime) * node_vars[i - 1][node1.index]) for
                        node1 in self.inn.nodes[i - 1]) + self.inn.biases[node].get_bound(self.y_prime),
                                         name="forward_pass_output_node_" + str(node))
                    # add robust counterfactual constraint:
                    if self.y_prime:
                        self.model.addConstr(node_var[node.index] - self.eps >= 0.0, name="output_node_lb_>=0")
                    else:
                        self.model.addConstr(node_var[node.index] + self.eps <= 0.0, name="output_node_ub_<0")
                    self.model.update()
            node_vars[i] = node_var
            if i != (self.inn.num_layers - 1):
                aux_vars[i] = aux_var
        return node_vars, aux_vars

    def create_constraints(self):
        node_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        aux_vars = dict()  # dict of {layer number, {Node's idx int, Gurobi variable obj}}
        node_vars[0] = self.add_input_variable_constraints()
        node_vars, aux_vars = self.add_node_variables_constraints(node_vars, aux_vars)
        return node_vars, aux_vars

    def set_objective(self, node_vars):
        obj_vars = []
        for idx in node_vars[0].keys():
            this_obj_var = self.model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"obj_xd_{idx}")
            self.model.update()
            self.model.addConstr(this_obj_var >= (self.x[idx] - node_vars[0][idx]), name="objective1")
            self.model.addConstr(this_obj_var >= (node_vars[0][idx] - self.x[idx]), name="objective2")
            self.model.update()
            obj_vars.append(this_obj_var)
        self.model.setObjective(quicksum(obj_vars), GRB.MINIMIZE)
        self.model.update()

    def set_objective_l1_l0(self, node_vars):
        obj_vars_l1 = []  # each is l1 distance for 1 feature
        obj_vars_l0 = []  # each is l0 distance for 1 feature
        for feat_idx in range(self.dataset.num_features):
            self.model.update()
            this_obj_var_l1 = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name=f"objl1_feat_{feat_idx}")
            this_obj_var_l0 = self.model.addVar(vtype=GRB.BINARY, name=f"objl0_feat_{feat_idx}")
            var_idxs = self.dataset.feat_var_map[feat_idx]

            if self.dataset.feature_types[feat_idx] == DataType.DISCRETE:
                to_max = dict()
                for var_idx in var_idxs:
                    to_max[var_idx] = self.model.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY,
                                                        name=f"objl1_feat_disc_{var_idx}")
                    self.model.update()
                    self.model.addConstr(to_max[var_idx] >= (node_vars[0][var_idx] - self.x[var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=-")
                    self.model.addConstr(to_max[var_idx] >= (self.x[var_idx] - node_vars[0][var_idx]),
                                         name=f"objl1_feat_disc_{var_idx}_>=+")     # for abs()
                    self.model.update()
                self.model.addConstr(this_obj_var_l1 == max_([to_max[idx] for idx in to_max.keys()]),
                                     name=f"objl1_feat_{feat_idx}")
                self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                                     name=f"objl0_feat_{feat_idx}")  # if l1<= 0, l0=0

            if self.dataset.feature_types[feat_idx] == DataType.ORDINAL:
                self.model.addConstr(this_obj_var_l1 >= (
                            quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])) / (
                                                 len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(this_obj_var_l1 >= (
                            np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])) / (
                                             len(var_idxs) - 1), name=f"objl1_feat_{feat_idx}_>=+")     # for abs()
                self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                                     name=f"objl0_feat_{feat_idx}")

            if self.dataset.feature_types[feat_idx] == DataType.CONTINUOUS_REAL:
                self.model.addConstr(
                    this_obj_var_l1 >= (quicksum([node_vars[0][idx] for idx in var_idxs]) - np.sum(self.x[var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=-")
                self.model.addConstr(
                    this_obj_var_l1 >= (np.sum(self.x[var_idxs]) - quicksum([node_vars[0][idx] for idx in var_idxs])),
                    name=f"objl1_feat_{feat_idx}_>=+")      # for abs()
                self.model.addConstr((this_obj_var_l0 == 0) >> (this_obj_var_l1 <= self.eps),
                                     name=f"objl0_feat_{feat_idx}")

            obj_vars_l1.append(this_obj_var_l1)
            obj_vars_l0.append(this_obj_var_l0)

        #self.model.setObjective(
        #    quicksum(obj_vars_l1) / self.dataset.num_features + quicksum(obj_vars_l0) / self.dataset.num_features,
        #    GRB.MINIMIZE)
        self.model.setObjective(quicksum(obj_vars_l1) / self.dataset.num_features, GRB.MINIMIZE)    # only L1
        self.model.update()

    def compute(self):
        node_vars, aux_vars = self.create_constraints()
        self.set_objective_l1_l0(node_vars)
        self.model.Params.LogToConsole = 0  # disable console output
        self.model.optimize()
        xp = []
        try:
            for v in self.model.getVars():
                if 'x_' in v.varName:
                    xp.append(v.getAttr(GRB.Attr.X))
            self.x_prime = np.array(xp)
        except:
            xp = None
        if xp is not None:
            return self.x_prime
        else:
            return None

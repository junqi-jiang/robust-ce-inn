import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neural_network import MLPClassifier
from inn import *


# util
def get_flattened_weight_and_bias(clf, weights=True, biases=True):
    if not weights and not biases:
        return 0
    w_concat = []
    b_concat = []
    if weights:
        w_all = clf.coefs_
        w_concat = np.append(w_all[0].flatten(), w_all[1].flatten())
    if biases:
        b_all = clf.intercepts_
        b_concat = np.append(b_all[0].flatten(), b_all[1].flatten())
    wb_concat = np.append(w_concat, b_concat)
    return wb_concat


def inf_norm(x, y):
    return np.max(abs(x - y))


def inf_norm_percentage(x, y):
    return np.max(abs(x - y) / abs(x))


def calculate_avg_2_dist(hidden_layer_size_val, X_train, y_train):
    training_sizes = [int(i * len(X_train)) for i in np.arange(0.5, 1.01, 0.05)]
    clfs = []
    wbs = []  # flattened concatenated weights and biases
    for size in training_sizes:
        this_clf = MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=hidden_layer_size_val,
                                 learning_rate_init=0.01, batch_size=32, max_iter=5000, random_state=11)
        this_clf.fit(X_train[:size], y_train[:size])
        clfs.append(this_clf)
        wbs.append(get_flattened_weight_and_bias(this_clf))

    # calculate p=2-distance
    dists_wbs = []
    dists_wbs_norm = []
    for i in range(1, len(wbs)):
        dists_wbs.append(round(euclidean_distances(wbs[i - 1].reshape(1, -1), wbs[i].reshape(1, -1)).flatten()[0], 3))
        dists_wbs_norm.append(round(euclidean_distances((wbs[i - 1] / wbs[0].shape[0]).reshape(1, -1),
                                                        (wbs[i] / wbs[0].shape[0]).reshape(1, -1)).flatten()[0], 3))

    avg_dist = np.mean(dists_wbs)
    avg_dist_norm = np.mean(dists_wbs_norm)
    return avg_dist, avg_dist_norm


def incremental_train(train_gap, base_clf, X_train, y_train, ignore_first_model=False, percentage_normalise=True):
    """
    function for incremental training on D2
    :param train_gap: increment when training on D2. 0.05 means train 5%, 10%, ..., 95%, 100% on D2
    :param base_clf: NN trained on D1
    :param X_train: training set of D2
    :param y_train: same
    :param ignore_first_model: ignore the change between the base model and the first incremented model
    :param percentage_normalise: if True, calculate percentage change of weights instead of absolute change
    :return: inf-distance of weights and biases
    """
    # training_sizes = [int(i * len(X_train)) for i in (np.arange(0, 1.01, train_gap))[1:]]
    training_sizes = [int(i * len(X_train)) for i in (np.arange(0, 1.01, train_gap))]
    ws = []  # flattened weights for every classifier in the range
    bs = []

    this_clf = copy.copy(base_clf)
    ws.append(get_flattened_weight_and_bias(this_clf, True, False))
    bs.append(get_flattened_weight_and_bias(this_clf, False, True))

    # incremental training
    for i, size in enumerate(training_sizes):
        if i == 0:
            continue
        inc_idx_l = training_sizes[i - 1] + 1
        inc_idx_r = training_sizes[i]
        this_clf = this_clf.partial_fit(X_train[inc_idx_l:inc_idx_r], y_train[inc_idx_l:inc_idx_r], [0, 1])
        ws.append(get_flattened_weight_and_bias(this_clf, True, False))
        bs.append(get_flattened_weight_and_bias(this_clf, False, True))

    # calculate inf-distance
    dists_ws = []
    dists_bs = []

    for i in range(1, len(ws)):
        if ignore_first_model and i == 1:
            continue
        if not percentage_normalise:
            dists_ws.append(round(inf_norm(ws[i - 1], ws[i]), 3))
            dists_bs.append(round(inf_norm(bs[i - 1], bs[i]), 3))
        else:
            dists_ws.append(round(inf_norm_percentage(ws[i - 1], ws[i]), 3))
            dists_bs.append(round(inf_norm_percentage(bs[i - 1], bs[i]), 3))
    avg_dist_w = np.mean(dists_ws)
    avg_dist_b = np.mean(dists_bs)

    return dists_ws, avg_dist_w, ws, dists_bs, avg_dist_b, bs


def plot_incremental_training_weight_bias_changes(training_gap, hidden_size, lr, batch_sz, max_iter, d1_x, d1_y, d2_x,
                                                  d2_y, percentage_normalise=False):
    avg_dists_w = []
    avg_dists_b = []
    h_ds_size = (np.arange(0, 1.01, training_gap))[1:]
    figw, axsw = plt.subplots(3, 3)
    figw.tight_layout(rect=[0, 0.1, 1, 0.95])
    figb, axsb = plt.subplots(3, 3)
    figb.tight_layout(rect=[0, 0.03, 1, 0.95])
    print(
        f"inf-distance for weights and biases of classifiers with hidden size {hidden_size}, training gap is {training_gap}, ")
    print("WEIGHTS or BIASES inf-distance /// % of D2 used")
    for i, h_size in enumerate(hidden_size):
        # for this hidden layer size, train a base clf
        this_clf = MLPClassifier(learning_rate='adaptive', hidden_layer_sizes=h_size, learning_rate_init=lr,
                                 batch_size=batch_sz,
                                 max_iter=max_iter, random_state=0)
        this_clf.fit(d1_x, d1_y)

        this_dists_w, this_avg_w, _, this_dists_b, this_avg_b, _ = incremental_train(training_gap, this_clf, d2_x, d2_y,
                                                                                     percentage_normalise=percentage_normalise)
        avg_dists_w.append(this_avg_w)
        avg_dists_b.append(this_avg_b)
        axsw[int(i / 3), i % 3].plot(h_ds_size, this_dists_w)
        axsw[int(i / 3), i % 3].set_title(f'#{h_size}, weights')
        axsb[int(i / 3), i % 3].plot(h_ds_size, this_dists_b)
        axsb[int(i / 3), i % 3].set_title(f'#{h_size}, biases')

    plt.figure(3)
    plt.plot(hidden_size, avg_dists_w, 'ro-')
    plt.title('avg WEIGHTS inf-distance /// hidden layer sizes')

    plt.figure(4)
    plt.plot(hidden_size, avg_dists_b, 'ro-')
    plt.title('avg BIASES inf-distance /// hidden layer sizes')
    plt.show()
    return avg_dists_w, avg_dists_b


def get_clf_num_layers(clf):
    if isinstance(clf.hidden_layer_sizes, int):
        return 3
    else:
        return len(clf.hidden_layer_sizes) + 2


def build_inn_nodes(clf, num_layers):
    nodes = dict()
    for i in range(num_layers):
        this_layer_nodes = []
        if i == 0:
            num_nodes_i = clf.n_features_in_
        elif i == num_layers - 1:
            num_nodes_i = 1
        else:
            if isinstance(clf.hidden_layer_sizes, int):
                num_nodes_i = clf.hidden_layer_sizes
            else:
                num_nodes_i = clf.hidden_layer_sizes[i-1]

        for j in range(num_nodes_i):
            this_layer_nodes.append(Node(i, j))
        nodes[i] = this_layer_nodes
    return nodes


def build_inn_weights_biases(clf, num_layers, delta, nodes):
    ws = clf.coefs_
    bs = clf.intercepts_
    weights = dict()
    biases = dict()
    for i in range(num_layers-1):
        for node_from in nodes[i]:
            for node_to in nodes[i+1]:
                w_val = ws[i][node_from.index][node_to.index]
                weights[(node_from, node_to)] = Interval(w_val, w_val-delta, w_val+delta)
                if i != 0:
                    b_val = bs[i-1][node_to.index]
                    biases[node_to] = Interval(b_val, b_val-delta, b_val+delta)
    return weights, biases


from model import *
import matplotlib.pyplot as plt
import numpy as np
from random import randint


def labels(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


# 2-var image graph
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def im_graph(agent1, agent2, x_label, y_label, title):
    scale1 = int(max(agent1)) + 1
    scale2 = int(max(agent2)) + 1
    low1 = int(min(agent1))
    low2 = int(min(agent2))
    num_cells = len(agent1)  # should be equal to length of perceived returns
    fig, ax = plt.subplots(figsize=(10, 10))
    im_graph = np.zeros((10, 10))
    for i in range(num_cells):
        x = int(10 * agent1[i] / scale1)  # int(round(agent_k[i],0))
        y = int(10 * agent2[i] / scale2)  # int(round(agent_perceived_return[i],0))
        print("x:",x,"   y:",y)
        im_graph[x][y] += 1
    ax.imshow(im_graph, cmap=plt.cm.Reds, interpolation='nearest', extent=[low1, scale1, low2, scale2])
    ax.set_aspect(0.5*scale1/scale2)
    labels(x_label, y_label, title)


# scatterplot
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def scatterplot(actual, perceived, x_label, y_label, title):
    plt.figure(randint(1000,9999))
    actual_numpy = np.asarray(flatten_list(actual))
    perceived_numpy = np.asarray(perceived)
    resid = actual_numpy-perceived_numpy
    time_id = []
    for np_array_idx in range(len(actual)):
        for i in range(len(actual[np_array_idx])):
            time_id.append(np_array_idx+2)  # index shift
    min_scale = int(min(resid))
    max_scale = int(max(resid))+1
    step = int((max_scale-min_scale)/10)+1
    plt.yticks(np.arange(min_scale, max_scale, step))
    plt.xticks(np.arange(0,max(time_id)+1,1))
    plt.scatter(time_id, resid)
    plt.axhline(0, color='black')
    labels(x_label, y_label, title)

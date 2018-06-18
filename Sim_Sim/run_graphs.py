from model import *
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math
import pandas as pd
from functions import *
import matplotlib as mpl


def settings():
    mpl.rcParams['xtick.labelsize'] = 5
    mpl.rcParams['ytick.labelsize'] = 5

# labeling x, y, and title
def labels(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


# 2-var image graph
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def im_graph(agent1, agent2, x_label, y_label, title, withZero, file_name, linear):
    settings()
    if linear:
        file_name += '_linear'
    else:
        # agent1 = log_0(np.asarray(agent1))
        agent2 = log_0(np.asarray(agent2))
        file_name += '_exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
    high1 = math.ceil(max(agent1))
    high2 = math.ceil(max(agent2))
    low1 = min(agent1)
    low2 = min(agent2)
    num_cells = len(agent1)  # should be equal to length of perceived returns
    fig, ax = plt.subplots(figsize=(high2-int(low2)+1, high1-int(low1)+1))  # swapping x and y has no effect on scatterplot
    im_graph = np.zeros((high2-int(low2)+1, high1-int(low1)+1))
    for i in range(num_cells):
        if not withZero and agent2[i] == 0:
            continue
        x = int(agent1[i] - low1)
        y = int(agent2[i] - low2)
        im_graph[high2-int(low2)-y][x] += 1
    plt.imshow(im_graph, cmap=plt.cm.Reds, interpolation='nearest', extent=[int(low1), high1, int(low2), high2])
    plt.colorbar()
    ax.set_aspect((high1-low1)/(high2-low2))
    labels(x_label, y_label, title)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1300.0 / float(DPI), 1220.0 / float(DPI))
    plt.savefig('web/images/imgraph_' + file_name)
    plt.close()
    del agent1, agent2, fig, ax, im_graph


# scatterplot that plots residuals of returns
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: actual array is still in numpy form, hasn't been flattened yet
# CONDITION: k array should be equal to length of perceived returns
def resid_scatterplot(actual, perceived, x_label, y_label, title):
    plt.figure(randint(1000,9999))
    actual_numpy = np.asarray(flatten_list(actual))
    perceived_numpy = np.asarray(perceived)
    resid = actual_numpy-perceived_numpy
    agent_id = []
    for np_array_idx in range(len(actual)):
        for i in range(len(actual[np_array_idx])):
            agent_id.append(np_array_idx+1)  # index shift
    min_scale = int(min(resid))
    max_scale = int(max(resid))+1
    step = int((max_scale-min_scale)/10)+1
    plt.yticks(np.arange(min_scale, max_scale, step))
    plt.xticks(np.arange(0, len(actual)+1, 1))
    plt.scatter(agent_id, resid)
    plt.axhline(0, color='black')
    labels(x_label, y_label, title)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches((2000.0+30*len(actual)) / float(DPI), 2000.0 / float(DPI))
    plt.savefig('web/images/scatterplot_resid')
    plt.close()
    del actual, perceived, actual_numpy, perceived_numpy, resid, agent_id, fig


# plots the returns vs cost on a scatterplot
def two_var_scatterplot(varx, vary, x_label, y_label, title, linear):  # , hline, vline):
    if linear:
        name = "linear"
        step_y = 10
    else:
        name = "exp"
        y_label = "Log of " + y_label
        title = "Log of " + title
        vary = log_0(vary)
        step_y = 1
    plt.figure(randint(1000,9999))
    max_y = max(vary)
    plt.scatter(varx, vary)
    labels(x_label, y_label, title)
    plt.yticks(np.arange(0, max_y+1, step_y))
    plt.xticks(np.arange(0, max(varx)+1, 1))
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(2000.0 / float(DPI), (2000.0+max_y) / float(DPI))
    plt.savefig('web/images/two_var_scatterplot_'+name)
    plt.close()
    del varx, vary, fig


# plots the young vs old scientist as a 2-var bar graph
def two_var_bar_graph(data, x_label, y_label, title, linear):
    if linear:
        name = "linear"
    else:
        name = "exp"
        y_label = "Log of " + y_label
        title = "Log of " + title
        data = [log_0(data[0]), log_0(data[1])]
    dict_data = {"Idea": range(0, len(data[0])), "Young": data[0], "Old": data[1]}
    df = pd.DataFrame.from_dict(dict_data)
    df.plot(kind='bar', stacked=True, width=1)
    df.plot.bar(x="Idea", y=["Young", "Old"])
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches((2000+10*len(data[0])) / float(DPI), 2000.0 / float(DPI))
    labels(x_label, y_label, title)
    plt.savefig('web/images/2-var_bar_graph_young_old_'+name)
    plt.close()
    del data, fig, dict_data, df

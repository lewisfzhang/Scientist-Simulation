# run_graphs.py

# workaround matplotlib bug (PNG)
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import math
import pandas as pd
from functions import *
from scipy.interpolate import spline
import timeit
import config
import pickle


# font settings for images
def font_settings(x):
    mpl.rcParams['xtick.labelsize'] = x
    mpl.rcParams['ytick.labelsize'] = x


# labeling x, y, and title
def labels(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


# 2-var image graph
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def im_graph(agent1, agent2, x_label, y_label, title, with_zero, file_name, linear):
    font_settings(10)
    if linear:
        file_name += '_linear'
    else:
        # agent1 = log_0(np.asarray(agent1))
        agent2 = log_0(agent2)
        file_name += '_exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
    high1 = math.ceil(max(agent1))
    high2 = math.ceil(max(agent2))
    low1 = min(agent1)
    low2 = min(agent2)
    num_cells = len(agent1)  # should be equal to length of perceived returns
    fig, ax = plt.subplots(figsize=(high2-int(low2)+1, high1-int(low1)+1))  # x and y interchangeable
    # same as initializing 1d np.zeros and then reshaping
    im_graph_data = np.zeros((high2-int(low2)+1, high1-int(low1)+1))
    for i in range(num_cells):
        if not with_zero and (agent2[i] == 0 or agent1[i] == 0):
            continue
        x = int(agent1[i] - low1)
        y = int(agent2[i] - low2)
        im_graph_data[high2-int(low2)-y][x] += 1
    plt.imshow(im_graph_data, cmap=plt.cm.Reds, interpolation='nearest', extent=[int(low1), high1, int(low2), high2])
    plt.colorbar()
    ax.set_aspect((high1-low1)/(high2-low2))
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/imgraph_' + file_name)
    with open(config.parent_dir + 'data/saved/imgraph_' + file_name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()
    del agent1, agent2, fig, ax, im_graph_data


# scatterplot that plots residuals of returns
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: actual array is still in numpy form, hasn't been flattened yet
# CONDITION: k array should be equal to length of perceived returns
def resid_scatterplot(actual, perceived, perceived_2d, x_label, y_label, title):
    plt.figure(5)
    font_settings(5)
    resid = np.asarray(actual)-np.asarray(perceived)
    agent_id = []
    for array_idx in range(len(perceived_2d)):
        for i in range(len(perceived_2d[array_idx])):
            agent_id.append(array_idx+1)  # index shift
    min_scale = int(min(resid))
    max_scale = int(max(resid))+1
    step = int((max_scale-min_scale)/10)+1
    plt.yticks(np.arange(min_scale, max_scale, step))
    plt.xticks(np.arange(1, len(perceived_2d)+1, 1))
    plt.scatter(agent_id, resid)
    plt.axhline(0, color='black')
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches((config.sq_width + 3 * len(actual)) / float(dpi), config.sq_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/scatterplot_resid')
    with open(config.parent_dir + 'data/saved/scatterplot_resid' + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()
    del actual, perceived, perceived_2d, resid, agent_id, fig


# plots the returns vs cost on a scatterplot
def two_var_scatterplot(varx, vary, x_label, y_label, title, linear):  # , hline, vline):
    plt.figure(6)
    font_settings(5)
    if linear:
        name = "linear"
        step_y = 10
    else:
        name = "exp"
        y_label = "Log of " + y_label
        title = "Log of " + title
        vary = log_0(vary)
        step_y = 1
    max_y = max(vary)
    plt.scatter(varx, vary)
    plt.yticks(np.arange(0, max_y+1, step_y))
    plt.xticks(np.arange(0, max(varx)+1, 1))
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.sq_width / float(dpi), (config.sq_width + max_y * 0.1) / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/two_var_scatterplot_'+name)
    with open(config.parent_dir + 'data/saved/two_var_scatterplot_' + name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()
    del varx, vary, fig


# plots the young vs old scientist as a 2-var bar graph
def two_var_bar_graph(data, x_label, y_label, title, linear):
    plt.figure(4)
    font_settings(5)
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
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches((config.sq_width + len(data[0])) / float(dpi), config.sq_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/2-var_bar_graph_young_old_'+name)
    with open(config.parent_dir + 'data/saved/2-var_bar_graph_young_old_' + name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()
    del data, fig, dict_data, df


# plots like a scatterplot but also has a line
# condition: y_var is a numpy array, not a list!
def line_graph(x_var, y_var, average, x_label, y_label, title, linear):
    plt.figure(1)
    font_settings(10)
    if average:
        name = "average_"
        y_var = divide_0(y_var, x_var)
    else:
        name = "total_"
    if linear:
        name += 'linear'
    else:
        name += 'exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
        y_var = log_0(y_var)
    x_var = np.arange(1, len(y_var))
    x_smooth = np.linspace(x_var.min(), x_var.max(), 200)
    y_smooth = spline(x_var, y_var[1:], x_smooth)
    plt.plot(x_smooth, y_smooth)
    plt.scatter(x_var, y_var[1:])
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/line_graph_'+name)
    with open(config.parent_dir + 'data/saved/line_graph_' + name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()


def one_var_bar_graph(data, legend, x_label, y_label, title, name):
    plt.figure(3)
    font_settings(10)
    x_var = np.arange(len(data))
    plt.bar(x_var, data, align='center', alpha=0.5, color='g')
    if legend is not None:
        plt.xticks(x_var, legend)
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/1-var_bar_graph_prop_'+name)
    with open(config.parent_dir + 'data/saved/1-var_bar_graph_prop_' + name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()
    del data, x_var


def two_var_line_graph(data, x_label, y_label, title, linear):
    plt.figure(2)
    font_settings(5)
    if linear:
        name = 'linear'
    else:
        name = 'exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
        data[0] = log_0(data[0])
        data[1] = log_0(data[1])
    x_var = np.arange(len(data[0]))
    x_smooth = np.linspace(x_var.min(), x_var.max(), 200)
    y_smooth_1 = spline(x_var, data[0], x_smooth)
    y_smooth_2 = spline(x_var, data[1], x_smooth)
    plt.plot(x_smooth, y_smooth_1, color='red', label='Young')
    plt.plot(x_smooth, y_smooth_2, color='blue', label='Old')
    plt.scatter(x_var, data[0], color='red')
    plt.scatter(x_var, data[1], color='blue')
    plt.legend()
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    plt.savefig(config.parent_dir + 'data/images/line_graph_'+name)
    with open(config.parent_dir + 'data/saved/2-var_line_graph_' + name + '.pkl', 'wb') as f:
        pickle.dump(fig, f)
    plt.close()

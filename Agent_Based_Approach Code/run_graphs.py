from model import *
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import math
import pandas as pd
from functions import *

def labels(x_label, y_label, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


# 2-var image graph
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def im_graph(agent1, agent2, x_label, y_label, title, withZero):
    print(agent2)
    high1 = math.ceil(max(agent1))
    high2 = math.ceil(max(agent2))
    low1 = int(min(agent1))
    low2 = int(min(agent2))
    num_cells = len(agent1)  # should be equal to length of perceived returns
    fig, ax = plt.subplots(figsize=(high2-low2+1, high1-low1+1))  # swapping x and y has no effect on scatterplot
    im_graph = np.zeros((high2-low2+1, high1-low1+1))
    for i in range(num_cells):
        if not withZero and agent2[i] == 0:
            continue
        x = int(agent1[i]-min(agent1))  # int(round(agent_k[i],0))
        y = int(agent2[i]-min(agent2))  # int(round(agent_perceived_return[i],0))
        im_graph[high2-low2-y][x] += 1
    plt.imshow(im_graph, cmap=plt.cm.Reds, interpolation='nearest', extent=[low1, high1, low2, high2])
    plt.colorbar()
    ax.set_aspect((high1-low1)/(high2-low2))
    labels(x_label, y_label, title)
    fig = plt.gcf()
    DPI = fig.get_dpi()
    fig.set_size_inches(1300.0 / float(DPI), 1220.0 / float(DPI))
    plt.savefig('web/images/graph' + str(page_counter()))

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
    plt.savefig('web/images/graph' + str(page_counter()))



def two_var_bar_graph(data, x_label, y_label, title):
    dict_data = {"Idea":range(0,len(data[0])),"Young":data[0],"Old":data[1]}
    # plt.figure(randint(1000,9999))
    df = pd.DataFrame.from_dict(dict_data)
    df.plot.bar(x="Idea", y=["Young", "Old"])
    plt.savefig('web/images/graph' + str(page_counter()))


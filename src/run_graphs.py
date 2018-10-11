# run_graphs.py

# workaround matplotlib bug (PNG)
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import math, config, pickle
import pandas as pd
from functions import *
from scipy.interpolate import spline
import functions as func
from matplotlib.ticker import MaxNLocator
from scipy import stats


# font settings for images
def font_settings(x):
    mpl.rcParams['xtick.labelsize'] = x
    mpl.rcParams['ytick.labelsize'] = x
    mpl.rcParams['legend.fontsize'] = 20


# labeling x, y, and title
def labels(x_label, y_label, title, font_size=25):
    plt.xlabel(x_label, fontsize=font_size)
    plt.ylabel(y_label, fontsize=font_size)
    plt.title(title, fontsize=font_size)


def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)
    plt.show()


def save_image(fig, name, in_tmp, step):
    if in_tmp:  # step should be an int if in_tmp is True
        path = config.tmp_loc+'step/step_'+str(step)+'/'
        func.create_directory(path)
        plt.savefig(path + name)
    else:
        plt.savefig(config.parent_dir + 'data/images/' + name)
        with open(config.parent_dir + 'data/saved/' + name + '.pkl', 'wb') as f:
            pickle.dump(fig, f)


# 2-var image graph
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: k array should be equal to length of perceived returns
def im_graph(agent1, agent2, x_label, y_label, title, with_zero, file_name, linear, in_tmp, step):
    font_settings(20)
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
    save_image(fig, 'imgraph_' + file_name, in_tmp, step)
    plt.close()
    del agent1, agent2, fig, ax, im_graph_data


# scatterplot that plots residuals of returns
# CONDITION: actual and perceived should have the same structure (arrangement of elements)
# CONDITION: actual array is still in numpy form, hasn't been flattened yet
# CONDITION: k array should be equal to length of perceived returns
def resid_scatterplot(actual, perceived, perceived_2d, x_label, y_label, title, in_tmp, step):
    plt.figure(5)
    font_settings(20)
    resid = np.asarray(actual)-np.asarray(perceived)
    agent_id = []
    for array_idx in range(len(perceived_2d)):
        for i in range(len(perceived_2d[array_idx])):
            agent_id.append(array_idx+1)  # index shift
    min_scale = int(min(resid))
    max_scale = int(max(resid))+1
    delta = int((max_scale-min_scale)/10)+1
    plt.yticks(np.arange(min_scale, max_scale, delta))
    plt.xticks(np.arange(1, len(perceived_2d)+1, 1))
    plt.scatter(agent_id, resid)
    plt.axhline(0, color='black')
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches((config.sq_width + 3 * len(actual)) / float(dpi), config.sq_width / float(dpi))
    save_image(fig, 'scatterplot_resid', in_tmp, step)
    plt.close()
    del actual, perceived, perceived_2d, resid, agent_id, fig


# plots the returns vs cost on a scatterplot
def two_var_scatterplot(varx, vary, x_label, y_label, title, linear, in_tmp, step):
    plt.figure(6)
    font_settings(20)
    count = 1
    if title[:2] == "2x":
        count = 2
        title = title[3:]+' (FUNDING)'  # format: "2x stuff"
    if linear:
        name = "linear"
        step_y = 10
    else:
        name = "exp"
        y_label = "Log of " + y_label
        title = "Log of " + title
        step_y = 1
    temp_varx = np.copy(varx)
    temp_vary = np.copy(vary)
    for i in range(count):
        plt.subplot(1, 2, i+1)
        if count != 1:
            varx = temp_varx[i]
            vary = temp_vary[i]
        if linear == "trend":
            if not linear:
                vary = log_0(vary)
            idx = np.argsort(varx)
            varx = varx[idx]
            vary = vary[idx]
            slope, intercept, r_value, p_value, std_err = stats.linregress(varx, vary)
            line = slope * varx + intercept
            plt.plot(varx, vary, 'o', varx, line)
            txt = "y={0}x+{1}\nr^2={2}".format(round(slope, 2), round(intercept, 2), round(r_value**2, 3))
            plt.text(0.5*(max(varx)+min(varx)), 0.9*(max(vary)+min(vary)), txt, fontsize=12)
            # # calc the trendline
            # z = np.polyfit(varx, vary, 3)  # rightmost number is the order of the polynomial
            # p = np.poly1d(z)
            # plt.plot(varx, p(vary), "r--")
            # # the line equation:
            # eq = "y=(%.6f)x^3+(%.6f)x^2+(%.6f)x+(%.6f)" % (z[0], z[1], z[2], z[3])
            # plt.text(0.4 * max(varx), 0.2 * max(vary), eq, fontsize=12)
        else:
            max_y = max(vary)
            plt.yticks(np.arange(0, max_y+1, step_y))
            plt.xticks(np.arange(0, max(varx)+1, 1))
        plt.scatter(varx, vary)
        labels(x_label, y_label, title)
        if count != 1:
            title = "(NO FUNDING)"
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = float(fig.get_dpi())*2
    fig.set_size_inches(count * config.sq_width / dpi, config.sq_width / dpi)
    save_image(fig, 'two_var_scatterplot_'+name, in_tmp, step)
    plt.close()
    del varx, vary, fig


# plots the young vs old scientist as a 2-var bar graph
def two_var_bar_graph(data, x_label, y_label, title, linear, in_tmp, step):
    plt.figure(4)
    font_settings(20)
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
    save_image(fig, '2-var_bar_graph_young_old_'+name, in_tmp, step)
    del data, fig, dict_data, df


# plots like a scatterplot but also has a line
# condition: y_var is a numpy array, not a list!
def line_graph(x_var, y_var, average, x_label, y_label, title, linear, in_tmp, step):
    plt.figure(1)
    font_settings(20)
    count = 1
    if title[:2] == "2x":
        count = 2
        title = title[3:]+' (FUNDING)'  # format: "2x stuff"
    if average is None:
        name = ''
    elif average:
        name = "average_"
    else:
        name = "total_"
    if linear is True:
        name += 'linear'
    elif linear is False:
        name += 'exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
    else:
        name += linear  # linear represents the string we want
    temp_x_var = np.copy(x_var)
    temp_y_var = np.copy(y_var)
    for i in range(count):
        if count != 1:
            y_var = temp_y_var[i]
        plt.subplot(1, 2, i+1)
        if average:
            x_var = temp_x_var[i]
            y_var = divide_0(y_var, x_var)
        if linear is False:
            y_var = log_0(y_var)
        if x_var is not None:
            x_var = np.arange(1, len(y_var))
        else:
            x_var = np.arange(len(y_var))
            y_var = np.insert(y_var, 0, -1)  # add a dummy -1 to beginning
        idx = np.where(np.absolute(y_var) > 0.01)[0][1:]
        x_var = x_var[idx-1]
        y_var = y_var[idx]
        # plt.scatter(x_var, y_var)
        if len(y_var) > 3:
            # blue spline line
            # x_smooth = np.linspace(x_var.min(), x_var.max(), 200)
            # y_smooth = spline(x_var, y_var, x_smooth)
            # plt.plot(x_smooth, y_smooth)

            # calc the trendline
            z = np.polyfit(x_var, y_var, 3)  # rightmost number is the order of the polynomial
            p = np.poly1d(z)
            plt.plot(x_var, p(x_var), "r--")
            # the line equation:
            eq = "y=(%.6f)x^3+(%.6f)x^2+(%.6f)x+(%.6f)" % (z[0], z[1], z[2], z[3])
            plt.text(0.4 * max(x_var), 0.2 * max(y_var), eq, fontsize=12)
        plt.xticks(np.arange(min(x_var), max(x_var) + 1, 2.0))
        labels(x_label, y_label, title)
        title = "NO FUNDING"
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(count * config.x_width / float(dpi), config.y_width / float(dpi))
    save_image(fig, 'line_graph_'+name, in_tmp, step)
    plt.close()


def one_var_bar_graph(data, legend, x_label, y_label, title, name, with_val, in_tmp, step):
    plt.figure(3)
    font_settings(20)
    count = 1
    if title[:2] == "2x":
        count = 2
        title = title[3:]+' (FUNDING)'  # format: "2x stuff"
    temp_data = np.copy(data)
    for i in range(count):
        plt.subplot(1, 2, i+1)
        if count != 1:
            data = temp_data[i]
        x_var = np.arange(len(data))
        plt.bar(x_var, data, align='center', alpha=0.5, color='g')
        if legend is not None:
            plt.xticks(x_var, legend)
        if with_val:
            for i, v in enumerate(data):
                plt.text(i - 0.03, v + 0.02, str(round(v, 2)), color='blue', fontweight='bold', fontsize=15)
        labels(x_label, y_label, title)
        if count != 1:
            title = "(NO FUNDING)"
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(count * config.x_width / float(dpi), config.y_width / float(dpi))
    save_image(fig, '1-var_bar_graph_prop_'+name, in_tmp, step)
    plt.close()
    del data


def two_var_line_graph(data, x_label, y_label, title, linear, in_tmp, step):
    plt.figure(2)
    font_settings(20)
    if linear is True:
        name = 'linear'
    elif linear is False:
        name = 'exp'
        y_label = "Log of " + y_label
        title = "Log of " + title
        data[0] = log_0(data[0])
        data[1] = log_0(data[1])
    else:
        name = linear  # linear represents the string we want
    mm = 1  # fixes width ratio for double plot
    if len(data) == 2:
        x_var = np.arange(len(data[0]))
        idx1 = np.where(np.absolute(data[0]) > 0.01)
        idx2 = np.where(np.absolute(data[1]) > 0.01)
        x_var1 = x_var[idx1]
        x_var2 = x_var[idx2]
        data0 = data[0][idx1]
        data1 = data[1][idx2]
        x_smooth_1 = np.linspace(x_var1.min(), x_var1.max(), 200)
        x_smooth_2 = np.linspace(x_var2.min(), x_var2.max(), 200)
        y_smooth_1 = spline(x_var1, data0, x_smooth_1)
        y_smooth_2 = spline(x_var2, data1, x_smooth_2)
        plt.plot(x_smooth_1, y_smooth_1, color='red', label='Young')
        plt.plot(x_smooth_2, y_smooth_2, color='blue', label='Old')
        plt.scatter(x_var1, data0, color='red')
        plt.scatter(x_var2, data1, color='blue')
    elif len(data) == 4:
        mm = 2
        key = ["(Funding)", "(No Funding)"]
        for i in range(2):
            plt.subplot(1, 2, i+1)
            j = i*2
            x_var = np.arange(len(data[j]))
            idx1 = np.where(np.absolute(data[j]) > 0.01)
            idx2 = np.where(np.absolute(data[j+1]) > 0.01)
            x_var1 = x_var[idx1]
            x_var2 = x_var[idx2]
            data0 = data[j][idx1]
            data1 = data[j+1][idx2]
            x_smooth_1 = np.linspace(x_var1.min(), x_var1.max(), 200)
            x_smooth_2 = np.linspace(x_var2.min(), x_var2.max(), 200)
            y_smooth_1 = spline(x_var1, data0, x_smooth_1)
            y_smooth_2 = spline(x_var2, data1, x_smooth_2)
            plt.plot(x_smooth_1, y_smooth_1, color='red', label='Young '+key[i])
            plt.plot(x_smooth_2, y_smooth_2, color='blue', label='Old '+key[i])
            plt.scatter(x_var1, data0, color='red')
            plt.scatter(x_var2, data1, color='blue')
            plt.legend()
            labels(x_label, y_label, title)
            title = ""
    elif len(data) == 5:
        mm = 2
        # with funding
        plt.subplot(1, 2, 1)
        plt.plot(data[0], data[1], color='red', label='Young (Funding)')
        plt.plot(data[0], data[2], color='blue', label='Old (Funding)')
        # plt.scatter(data[0], data[1], color='red')
        # plt.scatter(data[0], data[2], color='blue')
        plt.legend()
        labels(x_label, y_label, title)
        title = ""

        # with no funding
        plt.subplot(1, 2, 2)
        plt.plot(data[0], data[3], color='orange', label='Young (No Funding)')
        plt.plot(data[0], data[4], color='green', label='Old (No Funding)')
        # plt.scatter(data[0], data[3], color='orange')
        # plt.scatter(data[0], data[4], color='green')
    elif len(data) == 6:
        title += " (FUNDING)"
        mm = 2
        # with funding
        plt.subplot(1, 2, 1)
        idx = np.argsort(data[0])  # sort by index of x from least to greatest
        data[0] = data[0][idx]
        data[1] = data[1][idx]
        data[2] = data[2][idx]
        x_var = process_bin(data[0], len(data[0])//10, 10)
        y_1 = process_bin(data[1], len(data[1])//10, 10)
        y_2 = process_bin(data[2], len(data[2])//10, 10)
        plt.plot(x_var, y_1, color='red', label='Young (Funding)')
        plt.plot(x_var, y_2, color='blue', label='Old (Funding)')
        # plt.scatter(data[0], data[1], color='red')
        # plt.scatter(data[0], data[2], color='blue')
        plt.legend()
        labels(x_label, y_label, title)
        title = "(NO FUNDING)"

        # with no funding
        plt.subplot(1, 2, 2)
        idx = np.argsort(data[3])  # sort by index of x from least to greatest
        data[3] = data[3][idx]
        data[4] = data[4][idx]
        data[5] = data[5][idx]
        x_var = process_bin(data[3], len(data[3])//10, 10)
        y_1 = process_bin(data[4], len(data[4])//10, 10)
        y_2 = process_bin(data[5], len(data[5])//10, 10)
        plt.plot(x_var, y_1, color='orange', label='Young (No Funding)')
        plt.plot(x_var, y_2, color='green', label='Old (No Funding)')
        # plt.scatter(data[3], data[4], color='orange')
        # plt.scatter(data[3], data[5], color='green')
    plt.legend()
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(mm * config.x_width / float(dpi), config.y_width / float(dpi))
    save_image(fig, '2-var_line_graph_'+name, in_tmp, step)
    plt.close()


# plots like a scatterplot but also has a line
# condition: y_var is a numpy array, not a list!
def discrete_line_graph(y_var, x_label, y_label, title, name, in_tmp, step):
    plt.figure(5)
    font_settings(20)
    count = 1
    if title[:2] == "2x":
        count = 2
        title = title[3:]+' (FUNDING)'  # format: "2x stuff"
    temp_yvar = np.copy(y_var)
    for i in range(count):
        plt.subplot(2, 1, i+1)
        if count != 1:
            y_var = temp_yvar[i]
        x_var = np.arange(len(y_var))
        plt.scatter(x_var, y_var)
        plt.plot(x_var, y_var, "r--", linewidth=1)
        plt.text(0.1 * max(x_var), 0.9, "proportion formula:\ntotal effort / inv_logistic_cdf(0.99)", fontsize=12)
        labels(x_label, y_label, title)
        if count != 1:
            title = "(NO FUNDING)"
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches((config.x_width + len(y_var)*10) / float(dpi), count * config.y_width / float(dpi))
    save_image(fig, 'line_graph_'+name, in_tmp, step)
    plt.close()


def scatter_graph(data, x_label, y_label, title, name, in_tmp=False, step=None):
    # data[0] = x, data[1]=y
    plt.figure(7)
    font_settings(20)
    plt.scatter(data[0], data[1])
    m, b = np.polyfit(data[0], data[1], 1)
    plt.text(max(data[0])/2, 0.9, "trend line: y={0}x+{1}".format(m, b))
    graph("{0}*x+{1}".format(m, b), range(0, int(max(data[0]+1))))
    print("magic: {0}".format(1/m))
    labels(x_label, y_label, title)
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches((config.sq_width + 10*len(data[0])) / float(dpi), (config.sq_width + 10*len(data[1])) / float(dpi))
    save_image(fig, "scatter_"+name, in_tmp, step)
    plt.close()


def scatter_2_trial_graph(data, x_label, y_label, title, name, in_tmp=False, step=None):
    plt.figure(8)
    font_settings(20)
    data[1] = log_0(data[1])
    data[2] = log_0(data[2])
    idx1 = np.where(np.absolute(data[1]) > 0.01)
    idx2 = np.where(np.absolute(data[2]) > 0.01)
    p1 = plt.scatter(data[0][idx1], data[1][idx1], color='blue')
    p2 = plt.scatter(data[0][idx2], data[2][idx2], color='red')
    labels(x_label, y_label, title)
    plt.legend((p1, p2), ('Young', 'Old'))
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.sq_width / float(dpi), config.sq_width / float(dpi))
    save_image(fig, "scatter2_"+name, in_tmp, step)
    plt.close()


def stack_bar_graph(data, legend, x_label, y_label, title, name, with_val, in_tmp, step):
    plt.figure(9)
    font_settings(20)
    x_var = np.arange(len(data))
    p1 = plt.bar(x_var, data[:, 0], align='center', alpha=0.5, color='g')
    p2 = plt.bar(x_var, data[:, 1], bottom=data[:, 0], align='center', alpha=0.5, color='orange')
    if legend is not None:
        plt.xticks(x_var, legend)
    if with_val:
        for i, v in enumerate(data):
            plt.text(i - 0.02, v[0] + 0.003, str(round(v[0], 2)), color='blue', fontweight='bold', fontsize=15)
            plt.text(i - 0.02, v[0] + v[1] + 0.003, str(round(v[1], 2)), color='blue', fontweight='bold', fontsize=15)
            # for x in range(2):
            #     plt.text(i - 0.03, v[x] + 0.01, str(round(v[x], 2)), color='blue', fontweight='bold', fontsize=15)
            #     print(v[x], str(round(v[x], 2)), i - 0.03)
    labels(x_label, y_label, title)
    plt.legend((p1[0], p2[0]), ('Young', 'Old'))
    fig = plt.gcf()
    dpi = fig.get_dpi()
    fig.set_size_inches(config.x_width / float(dpi), config.y_width / float(dpi))
    save_image(fig, 'stack_bar_graph_prop_'+name, in_tmp, step)
    plt.close()
    del data, x_var, p1, p2


def double_bar_graph(data, legend, x_label, y_label, title, name, in_tmp=False, step=None):
    plt.figure(10)
    font_settings(20)
    x_var = np.arange(len(data))
    ax = plt.subplot(111)
    mm = 1
    if len(data[0]) == 2:
        p1 = ax.bar(x_var - 0.1, data[:, 0], width=0.2, color='g', align='center')
        p2 = ax.bar(x_var + 0.1, data[:, 1], width=0.2, color='orange', align='center')
        plt.legend((p1, p2), ('Young', 'Old'))
    elif len(data[0]) == 4:
        mm = 2
        plt.subplot(1, 2, 1)
        p1 = plt.bar(x_var - 0.1, data[:, 0], width=0.2, color='g', align='center')
        p2 = plt.bar(x_var + 0.1, data[:, 1], width=0.2, color='orange', align='center')
        plt.legend((p1, p2), ('Young (Funding)', 'Old (Funding)'))
        labels(x_label, y_label, title)
        title = ""
        if legend is not None:
            plt.xticks(x_var, legend)

        plt.subplot(1, 2, 2)
        p3 = plt.bar(x_var - 0.1, data[:, 2], width=0.2, color='red', align='center')
        p4 = plt.bar(x_var + 0.1, data[:, 3], width=0.2, color='blue', align='center')
        plt.legend((p3, p4), ('Young (No Funding)', 'Old (No Funding)'))
        labels(x_label, y_label, title)
    if legend is not None:
        plt.xticks(x_var, legend)
    fig = plt.gcf()
    dpi = float(fig.get_dpi()) * 2
    fig.set_size_inches(mm * config.sq_width / dpi, config.sq_width / dpi)
    save_image(fig, "2bar_"+name, in_tmp, step)
    plt.close()

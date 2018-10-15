# collect.py
# execute the run.py program before running collect.py!

import warnings as w
w.filterwarnings("ignore", message="numpy.dtype size changed")
w.filterwarnings("ignore", message="numpy.ufunc size changed")
import multiprocessing as mp
from run_graphs import *
import time, sys
import subprocess as s
from IPython.core.display import HTML


def main2():
    config.start = timeit.default_timer()
    use_mp = True

    # initiate multiprocessing with 'num_processors' threads
    # NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
    # slow down the program due to the additional overhead needed for process switching
    # NOTE: fork doesn't work on Mac, spawn is best because it works on Mac and is default on Windows
    if use_mp:
        try:
            mp.set_start_method('spawn')
        except Exception as e:
            print(e)  # context probably already set

        p = mp.Pool(processes=config.num_processors)  # default number is mp.cpu_count()

    # get starting time from run.py
    start_prog = int(open(config.tmp_loc + 'start_prog.txt', 'r').read())

    model_directory1 = '../zipped_archives/master/funding/'
    model_directory2 = '../zipped_archives/master/no_funding/'

    # loading variables after model is done running
    social_output = np_load2('social_output', model_directory1, model_directory2, is_same="double")
    ideas_entered = np_load2('ideas_entered', model_directory1, model_directory2, is_same="double")
    prop_age = np_load2('prop_age', model_directory1, model_directory2, is_same="double")
    prop_idea = np_load2('prop_idea', model_directory1, model_directory2, is_same="double")
    prop_idea_age = insert_x(np_load2('prop_idea_age', model_directory1, model_directory2, is_same=True))
    marginal_effort_by_age = np_load2('marginal_effort_by_age', model_directory1, model_directory2)
    idea_phase = np_load2("idea_phase", model_directory1, model_directory2, is_same="double")
    idea_phase_age = np_load2('idea_phase_age', model_directory1, model_directory2, is_same=True)
    prop_invested = np_load2("prop_invested", model_directory1, model_directory2, is_same="double")
    idea_qual = np_load2("idea_qual", model_directory1, model_directory2, is_same="double")
    age_effort_length = np_load2("age_effort_length", model_directory1, model_directory2)
    age_effort_time = np_load2("age_effort_time", model_directory1, model_directory2)
    age_effort_past = np_load2("age_effort_past", model_directory1, model_directory2)

    # print(social_output.shape, ideas_entered.shape, prop_age.shape, prop_idea.shape, prop_idea_age.shape, marginal_effort_by_age.shape,
    #       idea_phase.shape, idea_phase_age.shape, prop_invested.shape, idea_qual.shape, age_effort_length.shape, age_effort_time.shape,
    #       age_effort_past.shape)

    arg_list = [["line_graph", ideas_entered, social_output, True, "# of ideas entered in lifetime",
                 "total research output", "2x Average Total Research Output Given Number\nOf Ideas Entered in Lifetime", True],

                ["two_var_line_graph", marginal_effort_by_age, "Age of Idea\n(in TP's since idea's discovery)", "Total Marginal Effort",
                 "Effort Invested Given Idea Age", "marg"],

                ["line_graph", None, prop_age, None, "scientist age", "fraction paying k",
                 "2x Proportion of Scientists Paying to Learn By Age", "prop_age"],

                ["line_graph", None, prop_idea, None, "Age of Idea\n(in TP's since idea's discovery)", "Proportion of Scientists Working on the Idea",
                 "2x Proportion of Scientists Working Based on Age of An Idea", "prop_idea"],

                # ["line_graph", None, get_pdf(ideas_entered), None, "# of ideas entered in lifetime",
                #  "fraction working on 'x' ideas", "2x Proportion of Scientists Working on An Idea (PDF)", "prop_ideas_pdf"],

                # ["one_var_bar_graph", idea_phase, ["Investment", "Explosion", "Senescence"], "idea phases",
                #  "proportion of ideas worked on", "2x # of ideas worked on per idea phase", "idea_phase", True],

                # ["discrete_line_graph", prop_invested, "ideas", "prop invested",
                #  "2x Distribution of Social Returns Invested Across Ideas", "prop_invested"],

                ["two_var_scatterplot", idea_qual, prop_invested,
                 "idea quality\n(based on M, the slope multiplier)", "% exhausted\n(same as prop invested)",
                 "2x Distribution of Social Returns Invested\nBased on Idea Quality\n", "trend"],

                ["two_var_line_graph", age_effort_length, "Investment Style of Idea\n(Short-Term ----> Long-Term)",
                 "Average Effort Invested By All Scientists", "Effort Invested Into Idea Based on Investment Style",
                 "age_effort_magnitude"],

                ["two_var_line_graph", age_effort_time, "Age of Idea\n(in TP's since idea's discovery)",
                 "Average Effort Invested By All Scientists", "Effort Invested Given Idea Age",
                 "age_effort_time"],

                ["two_var_line_graph", age_effort_past, "Age of Idea\n(Based on Amount of Past Investment)",
                 "Average Effort Invested By All Scientists", "Effort Invested Given Idea Age",
                 "age_effort_past"],

                ["two_var_line_graph", prop_idea_age, "Age of Idea\n(in TP's since idea's discovery)", "Proportion of Scientists Working on the Idea",
                 "Proportion of Scientists Working Based on Age of Idea", "idea_age"],

                ["double_bar", idea_phase_age, ["Investment", "Explosion", "Senescence"], "idea phases",
                 "proportion of ideas worked on", "Distribution of Effort into Ideas By Phase", "idea_phase_age"]]

    for i in arg_list:
        i.append(False)
        i.append(False)
        if not use_mp:
            # print('\n\nstarting', i)
            func_distr(*i)

    if use_mp:
        p.starmap(func_distr, arg_list)  # starmap maps each function call into a parallel thread
        p.close()
        p.join()

    # saves all of the images to an html file
    png_to_html(path)

    stop_run("Total time to process data")
    f_print("\nEND OF PROGRAM\ntotal runtime:", time.time() - start_prog, "seconds\n\n")


def main(in_tmp, step, path):
    config.start = timeit.default_timer()

    # initiate multiprocessing with 'num_processors' threads
    # NOTE: increasing the number of processors does not always increase speed of program. in fact, it may actually
    # slow down the program due to the additional overhead needed for process switching
    # NOTE: fork doesn't work on Mac, spawn is best because it works on Mac and is default on Windows
    try:
        mp.set_start_method('spawn')
    except Exception as e:
        print(e)  # context probably already set

    p = mp.Pool(processes=config.num_processors)  # default number is mp.cpu_count()

    # get starting time from run.py
    start_prog = int(open(config.tmp_loc + 'start_prog.txt', 'r').read())

    # loading variables after model is done running
    model_directory = config.tmp_loc + 'model/'
    agent_vars = pd.read_pickle(model_directory + 'agent_vars_df.pkl')
    model_vars = pd.read_pickle(model_directory + 'model_vars_df.pkl')
    ideas = pd.read_pickle(model_directory + 'ideas.pkl')
    ind_ideas = pd.read_pickle(model_directory + 'ind_ideas.pkl')
    effort_invested_by_age = np.load(model_directory + 'effort_invested_by_age.npy')
    social_output = np.load(model_directory + 'social_output.npy')
    ideas_entered = np.load(model_directory + 'ideas_entered.npy')
    prop_age = np.load(model_directory + 'prop_age.npy')
    prop_idea = np.load(model_directory + 'prop_idea.npy')
    prop_idea_age = np.load(model_directory + 'prop_idea_age.npy')
    marginal_effort_by_age = np.load(model_directory + 'marginal_effort_by_age.npy')
    idea_phase = np.load(model_directory + 'idea_phase.npy')
    idea_phase_age = np.load(model_directory + 'idea_phase_age.npy')
    prop_remaining = np.load(model_directory + 'prop_remaining.npy')
    prop_invested = np.load(model_directory + 'prop_invested.npy')
    pVSa = np.load(model_directory + 'pVSa.npy')
    age_effort_length = np.load(model_directory + 'age_effort_length.npy')
    age_effort_time = np.load(model_directory + 'age_effort_time.npy')
    age_effort_past = np.load(model_directory + 'age_effort_past.npy')
    with open(model_directory + "final_perceived_returns_invested_ideas.txt", "rb") as fp:
        final_perceived_returns_invested_ideas = pickle.load(fp)

    arg_list = [["agent", agent_vars], ["model", model_vars], ["ideas", ideas],  # ["ind_ideas", ind_ideas],

                # ["im_graph", ind_ideas['agent_k_invested_ideas'], ind_ideas['agent_perceived_return_invested_ideas'],
                #  "k", "perceived returns",
                #  "perceived return vs cost for all INVESTED ideas across all scientists,time periods (biased)", False,
                #  "perceived", True],
                #
                ["im_graph", ind_ideas['agent_k_invested_ideas'], ind_ideas['agent_perceived_return_invested_ideas'],
                 "k", "perceived returns",
                 "perceived return vs cost for all INVESTED ideas across all scientists,time periods [biased)", False,
                 "perceived", False],
                #
                # ["im_graph", ind_ideas['agent_k_invested_ideas'], ind_ideas['agent_actual_return_invested_ideas'],
                #  "k", "actual returns",
                #  "actual return vs cost for all INVESTED ideas across all scientists,time periods [biased)", False,
                #  "actual", True],
                #
                ["im_graph", ind_ideas['agent_k_invested_ideas'], ind_ideas['agent_actual_return_invested_ideas'],
                 "k", "actual returns",
                 "actual return vs cost for all INVESTED ideas across all scientists,time periods [biased)", False,
                 "actual", False],
                #
                # # COMMENTED OUT PARAMS ARE GRAPHS THAT PLOT FOR EACH INDIVIDUAL SCIENTIST THAT AREN"T WORTH GRAPHING
                # # (they take a lot of time to graph since there's so many scientists but they don't tell use anything)
                #
                ["resid_scatterplot", ind_ideas['agent_actual_return_invested_ideas'],
                 ind_ideas['agent_perceived_return_invested_ideas'], final_perceived_returns_invested_ideas,
                 "Scientist ID", "Residual", "Residuals for all INVESTED ideas [actual-perceived)"],
                #
                # ["two_var_bar_graph", effort_invested_by_age, "Idea", "Marginal Effort Invested",
                #  "Marginal Effort Invested By Young and Old Scientists For All Ideas", True],
                #
                # ["two_var_bar_graph", effort_invested_by_age, "Idea", "Marginal Effort Invested",
                #  "Marginal Effort Invested By Young and Old Scientists For All Ideas", False],
                #
                # # runtime is WAY too long for linear y
                # ["two_var_scatterplot", ideas['avg_k'], ideas['total_pr'], "k", "perceived returns",
                #  "perceived return vs cost for INVESTED ideas [plot to check for bias]", True],
                #
                # ["two_var_scatterplot", ideas['avg_k'], ideas['total_pr'], "k", "perceived returns",
                #  "perceived return vs cost for INVESTED ideas [plot to check for bias)", False],
                #
                # # puts the above scatterplot in perspective with other imgraphs
                # # this is for invested ideas across all scientists/tp while the other ones are just all the ideas that
                # # scientists invested in
                # ["im_graph", ideas['avg_k'], ideas['total_pr'], "k", "perceived returns",
                #  "(IM) perceived return vs cost for INVESTED ideas (plot to check for bias)", False, "IM", False],
                #
                # ["line_graph", ideas_entered, social_output, True, "# of ideas entered in lifetime",
                #  "total research output", "Average Total Research Output Vs # Of Ideas Entered in Lifetime", False],
                #
                # ["line_graph", ideas_entered, social_output, False, "# of ideas entered in lifetime",
                #  "total research output", "Cum Total Research Output Vs # Of Ideas Entered in Lifetime", False],
                #
                # ["line_graph", ideas_entered, social_output, False, "# of ideas entered in lifetime",
                #  "total research output", "Cum Total Research Output Vs # Of Ideas Entered in Lifetime", True],

                ["line_graph", ideas_entered, social_output, True, "# of ideas entered in lifetime",
                 "total research output", "Average Total Research Output Vs # Of Ideas Entered in Lifetime", True],

                ["two_var_line_graph", marginal_effort_by_age, "age of idea", "marginal effort",
                 "Effort Invested By Ages of Ideas and Scientists", False],

                ["one_var_bar_graph", prop_age, None, "scientist age", "fraction paying k",
                 "Proportion of Scientists Paying to Learn By Age", "age", True],

                ["one_var_bar_graph", prop_idea, None, "age of idea", "proportion of scientists working on the idea",
                 "Proportion of Scientists Working Based on Age of Idea", "idea", False],

                ["one_var_bar_graph", get_pdf(ideas_entered), None, "# of ideas entered in lifetime",
                 "fraction working on 'x' ideas", "Proportion of Scientists Working on An Idea (PDF)", "ideas_pdf", False],

                ["one_var_bar_graph", get_cdf(ideas_entered), None, "# of ideas entered in lifetime",
                 "fraction working on more than 'x' ideas", "Proportion of Scientists Working on An Idea (CDF)",
                 "ideas_cdf", False],

                ["one_var_bar_graph", idea_phase, ["Investment", "Explosion", "Senescence"], "idea phases",
                 "proportion of ideas worked on", "# of ideas worked on per idea phase", "idea_phase", True],

                ["discrete_line_graph", prop_invested, "ideas", "prop invested",
                 "Distribution of Social Returns Invested Across Ideas", "prop_invested"],

                ["discrete_line_graph", prop_remaining, "ideas", "prop remaining",
                 "Distribution of Social Returns Left Across Ideas", "prop_remaining"],

                # ["scatter", pVSa, "perceived", "actual", "Perceived VS Actual", 'pVSa'],
                #
                # ["stacked_bar", prop_idea_age, None, "age of idea", "proportion of scientists working on the idea",
                #  "Proportion of Scientists Working Based on Age of Idea (YOUNG VS OLD)", "idea_age", True],
                #
                # ["stacked_bar", idea_phase_age, ["Investment", "Explosion", "Old Age"], "idea phases",
                #  "proportion of ideas worked on", "# of ideas worked on per idea phase", "idea_phase_age", True],

                ["scatter_2", age_effort_length, "Short ----> Long\n(in terms of investment)",
                 "Effort Invested By All Scientists", "Young VS Old, Short VS Long Term Investments",
                 "age_effort_magnitude"],

                ["scatter_2", age_effort_time, "New ----> Old\n(in terms of age of idea)",
                 "Effort Invested By All Scientists", "Young VS Old, New VS Old Ideas",
                 "age_effort_time"],

                ["scatter_2", age_effort_past, "New ----> Old\n(in terms of amount of past investment)",
                 "Effort Invested By All Scientists", "Young VS Old, New VS Old Ideas",
                 "age_effort_past"],

                ["double_bar", prop_idea_age, None, "age of idea", "proportion of scientists working on the idea",
                 "Proportion of Scientists Working Based on Age of Idea (YOUNG VS OLD)", "idea_age"],

                ["double_bar", idea_phase_age, ["Investment", "Explosion", "Senescence"], "idea phases",
                 "proportion of ideas worked on", "# of ideas worked on per idea phase", "idea_phase_age"]]

    for i in arg_list:
        i.append(in_tmp)
        i.append(step)

    p.starmap(func_distr, arg_list)  # starmap maps each function call into a parallel thread
    p.close()
    p.join()

    # saves all of the images to an html file
    png_to_html(path)

    if step is None:
        stop_run("Total time to process data")
        f_print("\nEND OF PROGRAM\ntotal runtime:", time.time() - start_prog, "seconds\n\n")


# assigning which function to call in the run_graphs.py file
def func_distr(graph_type, *other):
    start = timeit.default_timer()

    # set dataframe settings to max width, max rows, and max columns since we are collecting large quantities
    # of data and printing out entire arrays/tuples
    pd.set_option("display.max_colwidth", -1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    if graph_type == "agent":  # in_tmp is None
        # agent dataframe (other[0] contains agent_vars)
        agent_vars = other[0]
        agent_vars = agent_vars.replace(np.nan, '', regex=True).replace("\\r\\n", "<br>", regex=True)
        HTML(agent_vars.to_html('../data/pages/page_agent_vars.html', escape=False))
        del agent_vars
    elif graph_type == "model":
        # model dataframe (other[0] contains model_vars)
        model_vars = other[0]
        model_vars = model_vars.replace(np.nan, '', regex=True).replace("\\r\\n", "<br>", regex=True)  # .transpose()
        HTML(model_vars.to_html('../data/pages/page_model_vars.html', escape=False))
        del model_vars
    elif graph_type == "ideas":
        # dataframe specifying info per idea
        data1 = other[0].astype(str)
        columns = ['scientists_invested', "times_invested", "avg_k", "total_effort (marginal)", "prop_invested",
                   "total_pr", "total_ar"]
        for col in columns:
            data1.ix[pd.to_numeric(data1[col], errors='coerce') == 0, [col]] = ''
        data1.to_html('../data/pages/page_ideas.html')
        del data1
    elif graph_type == "ind_ideas":
        ind_vars = other[0]
        ind_vars = ind_vars.transpose()
        ind_vars.to_html('../data/pages/page_ind_ideas.html')
        del ind_vars
    elif graph_type == "line_graph":
        line_graph(*other)
    elif graph_type == "im_graph":
        im_graph(*other)
    elif graph_type == "resid_scatterplot":
        resid_scatterplot(*other)
    elif graph_type == "two_var_scatterplot":
        two_var_scatterplot(*other)
    elif graph_type == "two_var_bar_graph":
        two_var_bar_graph(*other)
    elif graph_type == "one_var_bar_graph":
        one_var_bar_graph(*other)
    elif graph_type == "two_var_line_graph":
        two_var_line_graph(*other)
    elif graph_type == "discrete_line_graph":
        discrete_line_graph(*other)
    elif graph_type == 'scatter':
        scatter_graph(*other)
    elif graph_type == 'stacked_bar':
        stack_bar_graph(*other)
    elif graph_type == 'double_bar':
        double_bar_graph(*other)
    elif graph_type == 'scatter_2':
        scatter_2_trial_graph(*other)

    gc_collect()
    stop = timeit.default_timer()
    f_print("\nfinished", graph_type, stop-start, "seconds")


# helper method for main
def init(step=None):
    in_tmp = False
    path = None
    if step is not None:
        in_tmp = True
        path = config.tmp_loc + 'step/step_' + str(step) + '/'
    main(in_tmp, step, path)


def np_load2(name, loc1, loc2, is_same=False):
    loc1 += name + '.npy'
    loc2 += name + '.npy'
    loc1 = np.load(loc1)
    loc2 = np.load(loc2)
    if is_same == "double":  # when merging two 1D numpy arrays
        return np.asarray([loc1, loc2])
    if is_same == "vstack":
        return np.vstack([loc1, loc2])
    elif is_same:
        diff = len(loc1) - len(loc2)
        if diff < 0:  # loc 2 is greater, add to loc1
            diff *= -1
            loc1 = np.vstack([loc1, np.zeros(2*diff).reshape(diff, 2)])
        elif diff > 0:  # add to loc2
            loc2 = np.vstack([loc2, np.zeros(2*diff).reshape(diff, 2)])
        return np.hstack([loc1, loc2])
    else:
        return append_list(loc1, loc2)


# EXAMPLE
# array([[ 0,  4,  8, 12],
#        [ 1,  5,  9, 13],
#        [ 2,  6, 10, 14],
#        [ 3,  7, 11, 15]])
# BECOMES
# array([[ 2,  2,  2,  2],
#        [ 0,  1,  2,  3],
#        [ 4,  5,  6,  7],
#        [ 8,  9, 10, 11],
#        [12, 13, 14, 15]])
def insert_x(arr):
    return np.insert(arr.transpose(), 0, np.arange(len(arr)), axis=0)


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    # ensure current working directory is in src folder
    if os.getcwd()[-3:] != 'src':
        # assuming we are somewhere inside the git directory
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        print('changing working directory from', os.getcwd(), 'to', path)
        os.chdir(path + '/src')

    clear_images()

    in_tmp = False
    step = None
    path = None
    if len(sys.argv) > 1 and sys.argv[1] == 'master':
        main2()
    else:
        if len(sys.argv) == 2:
            in_tmp = True
            step = int(sys.argv[1])
            path = config.tmp_loc + 'step/step_' + str(step) + '/'

        main(in_tmp, step, path)

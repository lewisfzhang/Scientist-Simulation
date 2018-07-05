# run.py

from model import *
import config
import timeit
import time


def main():
    # start runtime
    start_prog = timeit.default_timer()

    create_directory(config.tmp_loc)
    with open(config.tmp_loc + 'start_prog.txt', 'w')as f:
        f.write('%d' % time.time())

    config.start = timeit.default_timer()

    # default parameters for model as a dictionary
    all_params = {"seed": config.seed, "use_multiprocessing": config.use_multiprocessing,
                  "use_store": config.use_store, "time_periods": config.time_periods,
                  "ideas_per_time": config.ideas_per_time, "N": config.N,
                  "time_periods_alive": config.time_periods_alive, "true_means_lam": config.true_means_lam,
                  "true_sds_lam": config.true_sds_lam, "start_effort_lam": config.start_effort_lam,
                  "k_lam": config.k_lam, "use_multithreading": config.use_multithreading}

    # initialize model object
    model = ScientistModel(config.seed)

    stop_run("time to create model object")

    # printing parameters into console screen
    print("\nVariables:\n", all_params)

    # write parameters to text file
    f = open('web/parameters.txt', 'w')
    f.write(str(all_params))
    f.close()

    stop_run("entering main function")
    gc_collect()

    for i in range(config.time_periods + 2):
        model.step()
        stop_run("step: "+str(i))

    print("\nTOTAL TIME TO FINISH RUNNING SIMULATION:", timeit.default_timer() - start_prog, "seconds")


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    main()

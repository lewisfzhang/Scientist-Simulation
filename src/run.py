# run.py

import sys
import init


def main():
    # if user wants to pass in arguments
    if len(sys.argv) == 5:
        init.time_periods = int(sys.argv[1])
        init.ideas_per_time = int(sys.argv[2])
        init.N = int(sys.argv[3])
        init.time_periods_alive = int(sys.argv[4])

    # so that config file loads after init.py is set
    import config
    import model as m
    import functions as func
    import timeit
    import time

    # start runtime
    start_prog = timeit.default_timer()

    func.create_directory(config.parent_dir + 'data/')
    func.create_directory(config.tmp_loc)
    with open(config.tmp_loc + 'start_prog.txt', 'w') as f:
        f.write('%d' % time.time())

    config.start = timeit.default_timer()

    # default parameters for model as a dictionary
    all_params = {"seed": config.seed, "use_multiprocessing": config.use_multiprocessing,
                  "use_store": config.use_store, "optimization": config.switch, "time_periods": config.time_periods,
                  "ideas_per_time": config.ideas_per_time, "N": config.N,
                  "time_periods_alive": config.time_periods_alive, "true_means_lam": config.true_means_lam,
                  "true_sds_lam": config.true_sds_lam, "start_effort_lam": config.start_effort_lam,
                  "k_lam": config.k_lam, "use_multithreading": config.use_multithreading}

    # initialize model object
    model = m.ScientistModel(config.seed)

    func.stop_run("time to create model object")

    # printing parameters into console screen
    print("\nVariables:\n", all_params)

    # write parameters to text file
    f = open('../data/parameters.txt', 'w')
    f.write(str(all_params))
    f.close()

    func.stop_run("entering main function")
    func.gc_collect()

    for i in range(config.time_periods + 2):
        model.step()
        func.stop_run("step: "+str(i))

    print("\nTOTAL TIME TO FINISH RUNNING SIMULATION:", timeit.default_timer() - start_prog, "seconds")


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    main()

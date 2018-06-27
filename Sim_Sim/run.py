# run.py

from model import *
import input_file
import timeit


def main():
    # start runtime
    start_prog = timeit.default_timer()
    with open('tmp/start_prog.txt', 'w')as f:
        f.write('%d' % start_prog)

    input_file.start = timeit.default_timer()

    # default parameters for model as a dictionary
    all_params = {"seed": input_file.seed, "use_multiprocessing": input_file.use_multiprocessing,
                  "use_multithreading": input_file.use_multithreading, "time_periods": input_file.time_periods,
                  "ideas_per_time": input_file.ideas_per_time, "N": input_file.N,
                  "true_means_lam": input_file.true_means_lam, "true_sds_lam": input_file.true_sds_lam,
                  "start_effort_lam": input_file.start_effort_lam, "noise_factor": input_file.noise_factor,
                  "k_lam": input_file.k_lam, "time_periods_alive": input_file.time_periods_alive}

    # initialize model object
    model = ScientistModel(input_file.seed)

    stop_run("time to create model object")

    # printing parameters into console screen
    print("\nVariables:\n", all_params)

    # write parameters to text file
    f = open('web/parameters.txt', 'w')
    f.write(str(all_params))
    f.close()

    stop_run("entering main function")
    gc_collect()

    for i in range(input_file.time_periods+2):
        model.step()
        stop_run("step: "+str(i))
        gc_collect()

    print("\nTOTAL TIME TO FINISH RUNNING SIMULATION:", timeit.default_timer() - start_prog, "seconds")


if __name__ == '__main__':  # for multiprocessor package so it knows the true main/run function
    main()

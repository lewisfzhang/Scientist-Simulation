# run.py

import sys, init, os, timeit, time
import subprocess as s


def main():
    with_collect = False  # whether or not to run collect after run is finished
    with_prompt = True

    if with_prompt:
        print("Are you sure you have checked the following variables?")
        print(" - with_collect (run.py)")
        print(" - has_past (pack_data.py)")
        check = input('(y/n) ')
        if not check in ['y', 'Y']:
            raise Exception("please check your variables!")
        print()

    # start runtime
    start_prog = timeit.default_timer()

    # ensure current working directory is in src folder
    if os.getcwd()[-3:] != 'src':
        # assuming we are somewhere inside the git directory
        path = s.Popen('git rev-parse --show-toplevel', shell=True, stdout=s.PIPE).communicate()[0].decode("utf-8")[:-1]
        print('changing working directory from', os.getcwd(), 'to', path)
        os.chdir(path + '/src')

    print('Args:',sys.argv[:])
    # if user wants to pass in arguments
    if len(sys.argv) >= 5:  # config 1
        init.time_periods = int(sys.argv[1])
        init.ideas_per_time = int(sys.argv[2])
        init.N = int(sys.argv[3])
        init.time_periods_alive = int(sys.argv[4])
    if len(sys.argv) >= 8:  # config 2
        init.prop_sds = float(sys.argv[5])
        init.prop_means = float(sys.argv[6])
        init.prop_start = float(sys.argv[7])
    if len(sys.argv) == 15:  # server config
        init.true_means_lam = float(sys.argv[5])
        init.prop_sds = float(sys.argv[6])
        init.prop_means = float(sys.argv[7])
        init.prop_start = float(sys.argv[8])
        init.switch = float(sys.argv[9])
        # sys.argv[10] is empty for now
        init.all_scientists = sys.argv[11] == 'True'
        init.use_equal = sys.argv[12] == 'True'
        init.use_idea_shift = sys.argv[13] == 'True'
        init.show_step = sys.argv[14] == 'True'

    # check if we are using batch runs
    if os.path.isdir('tmp_batch'):
        init.tmp_loc = 'tmp_batch/tmp_' + '_'.join([str(v) for v in sys.argv[1:]]) + '/'

    # so that config file loads after init.py is set
    import config, collect
    import model as m
    import functions as func

    func.create_directory(config.parent_dir + 'data/')
    func.create_directory(config.tmp_loc)
    with open(config.tmp_loc + 'start_prog.txt', 'w') as f:
        f.write('%d' % time.time())

    config.start = timeit.default_timer()

    # default parameters for model as a dictionary
    all_params = {"seed": config.seed, "use_multiprocessing": config.use_multiprocessing,
                  "use_store": config.use_store, "optimization": config.switch, "time_periods": config.time_periods,
                  "ideas_per_time": config.ideas_per_time, "N": config.N, "use_store_model": config.use_store_model,
                  "time_periods_alive": config.time_periods_alive, "true_means_lam": config.true_means_lam,
                  "true_sds_lam": config.true_sds_lam, "start_effort_lam": config.start_effort_lam,
                  "k_lam": config.k_lam, "use_multithreading": config.use_multithreading, "use_equal": config.use_equal,
                  "use_idea_shift": config.use_idea_shift}

    # printing parameters into console screen
    func.f_print("\nVariables:\n", all_params)

    # write parameters to text file
    f = open('../data/parameters.txt', 'w')
    f.write(str(all_params))
    f.close()

    # initialize model object
    model = m.ScientistModel(config.seed)

    func.stop_run("time to create model object... now entering main function")
    func.gc_collect()

    for i in range(config.time_periods + 2):
        model.step()
        func.stop_run("step: "+str(i))

    func.f_print("\nTOTAL TIME TO FINISH RUNNING SIMULATION:", timeit.default_timer() - start_prog, "seconds")

    if with_collect:
        s.call('python3 collect.py', shell=True)
    # collect.init()


if __name__ == '__main__':
    main()

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter
from mesa.batchrunner import BatchRunner

# can either use server to display interactive data (1 run), or do a batch of simultaneous runs
# use if we only care about model variables and how changing one variable affects the others
# NOTE: cannot access agent variables
if use_batch:
    fixed_params = {"time_periods": time_periods, "ideas_per_time": ideas_per_time, "N": N,
                    "max_investment_lam": max_investment_lam,
                    "true_sds_lam": true_sds_lam, "true_means_lam": true_means_lam,
                    "start_effort_lam": start_effort_lam,
                    "start_effort_decay": start_effort_decay, "noise_factor": noise_factor, "k_lam": k_lam,
                    "sds_lam": sds_lam,
                    "means_lam": means_lam, "seed": randint(100000, 999999)}
    # NOTE: variable should not be the range, because you should only run 1 iteration of it
    variable_params = {
        "time_periods_alive": time_periods_alive}  # [min,max) total number of values in array is (max-min)/step
    model_reports = {"Total_Effort": get_total_effort}

    batch_run = BatchRunner(ScientistModel,
                            fixed_parameters=fixed_params,
                            variable_parameters=variable_params,
                            iterations=5,
                            max_steps=time_periods + 2,
                            model_reporters=model_reports)
    batch_run.run_all()
    run_data = batch_run.get_model_vars_dataframe()
    # # NOTE: obviously this is stupid since we only have 5 steps, but this is a template for future batch runs
    # plt.scatter(run_data.max_investment_lam, run_data.Total_Effort)
    # plt.show()

# sliders allow us to change certain variables in real time
if use_slider:
    # sliders for ScientistModel(Model)
    time_periods = UserSettableParameter('slider', "Time Periods", 3, 1, 100, 1)
    ideas_per_time = UserSettableParameter('slider', "Ideas Per Time", 1, 1, 100, 1)
    N = UserSettableParameter('slider', "N", 2, 1, 100, 1)
    max_investment_lam = UserSettableParameter('slider', "Max Investment Lambda", 10, 1, 100, 1)
    true_sds_lam = UserSettableParameter('slider', "True SDS Lambda", 4, 1, 100, 1)
    true_means_lam = UserSettableParameter('slider', "True Means Lambda", 25, 1, 100, 1)

    # sliders for Scientist(Agent)
    start_effort_lam = UserSettableParameter('slider', "Start Effort Lambda", 10, 1, 100, 1)
    start_effort_decay = UserSettableParameter('slider', "Start Effort Decacy", 1, 1, 100, 1)
    k_lam = UserSettableParameter('slider', "K Lambda", 2, 1, 100, 1)
    sds_lam = UserSettableParameter('slider', "SDS Lambda", 4, 1, 100, 1)
    means_lam = UserSettableParameter('slider', "Means Lambda", 25, 1, 100, 1)

# launches an interactive display, probably don't need to implement this
# NOTE: not practical if we are running large scale simulations as calculations will take too long to keep
# up with the interactive display
if use_server:
    chart1 = ChartModule([{"Label": "Total Effort",
                           "Color": "Black"}],
                         data_collector_name='datacollector')
    server = ModularServer(ScientistModel,
                           [chart1],
                           "Scientist Model",
                           all_params)

    server.port = 8521  # the default
    server.launch()

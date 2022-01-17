import json
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
import inspect
import time
# ignore console warnings - comment this if you want to see the warning messages
import warnings
warnings.filterwarnings("ignore")

# access parent directory from notebooks directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import src.simulation as s
import src.utils as utils
print('Starting the ABM Script .. Loading Simulation')
input_args = utils.load_input_args('../input/simulation.json')
simulation =  s.Simulation()
pars_nomination = pd.read_csv('../output/opt_pars_nomination_test.csv', sep=',', header=0)

# NOMINATION NETWORKS 
list_results_nomm = []
list_results_avg_nomm = []
list_agents_per_intervention_nomm = []
start_whole = time.time()
count = 0

print('Starting the simulations...')

for index, row in pars_nomination.iterrows():
    results_nomm, results_avg_nomm, agents_per_intervention_nomm = simulation.simulate_interventions(700,'nomination',row['threshold'],row['ipa'])
    list_results_nomm.append(results_nomm)
    list_results_avg_nomm.append(results_avg_nomm)
    list_agents_per_intervention_nomm.append(agents_per_intervention_nomm)
    end = time.time()
    print(count, row['threshold'], row['ipa'], "Time elapsed:", end - start_whole, ' seconds')
    count = count + 1

print('Stimulations done! Postprocessing output...')

print('Step1: Create a list per run of overall mean PAL...' )
list_results_mean_nom = []
for run in range(len(list_results_avg_nomm)):
    # per run mean of all classes
    all_averaged = {}
    for i in input_args['intervention_strategy']:
        temp_res = pd.Series([], dtype = float)
        counter = 0
        for class_id,res in list_results_avg_nomm[run].items():
            temp_res = temp_res.add(list_results_avg_nomm[run][class_id][i],fill_value=0)
            counter = counter + 1
        all_averaged[i] = temp_res/counter

    list_results_mean_nom.append(all_averaged)

print('Step2: Create per intervention a list of 100...' )

out_indegree_nom = []
out_betweenness_nom = []
out_closeness_nom = []

for run in range(len(list_results_mean_nom)):
    # per run
    temp1 = list_results_mean_nom[run]
    out_indegree_nom.append(temp1['indegree'])
    out_betweenness_nom.append(temp1['betweenness'])
    out_closeness_nom.append(temp1['closeness'])

print('Step3: Mean and percentiles...' )
results_nom_indegree = pd.concat(out_indegree_nom, axis=1, keys=[s.name for s in out_indegree_nom])
results_nom_betweenness = pd.concat(out_betweenness_nom, axis=1, keys=[s.name for s in out_betweenness_nom])
results_nom_closeness = pd.concat(out_closeness_nom, axis=1, keys=[s.name for s in out_closeness_nom])

# indegree
i_nom = results_nom_indegree.mean(axis=1)
i_nom_min = results_nom_indegree.quantile(q=0.025, axis=1)
i_nom_max = results_nom_indegree.quantile(q=0.975, axis=1)

# betweenness
b_nom = results_nom_betweenness.mean(axis=1)
b_nom_min = results_nom_betweenness.quantile(q=0.025, axis=1)
b_nom_max = results_nom_betweenness.quantile(q=0.975, axis=1)

# closeness
c_nom = results_nom_closeness.mean(axis=1)
c_nom_min = results_nom_closeness.quantile(q=0.025, axis=1)
c_nom_max = results_nom_closeness.quantile(q=0.975, axis=1)


print('Step4: Success rates and confidence intervals...' )

i_nom_sr = (i_nom/i_nom[0] -1) *100
i_nom_sr_min = (i_nom_min/i_nom_min[0] -1) *100
i_nom_sr_max = (i_nom_max/i_nom_max[0] -1) *100
b_nom_sr = (b_nom/b_nom[0] -1) *100
b_nom_sr_min = (b_nom_min/b_nom_min[0] -1) *100
b_nom_sr_max = (b_nom_max/b_nom_max[0] -1) *100
c_nom_sr = (c_nom/c_nom[0] -1) *100
c_nom_sr_min = (c_nom_min/c_nom_min[0] -1) *100
c_nom_sr_max = (c_nom_max/c_nom_max[0] -1) *100

#Output to save - modify to discussed output
i_nom.to_csv('../output/indegree_nomination_avg.csv')
b_nom.to_csv('../output/betweenness_nomination_avg.csv')
c_nom.to_csv('../output/closeness_nomination_avg.csv')





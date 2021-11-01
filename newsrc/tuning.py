#from codes.networkPA import generate_network_PA
#from codes.simulatePA import diffuse_behavior_PA


import networkx as nx
import numpy as np
import json
import pandas as pd
import random
from pprint import pprint
from time import time

import newsrc.population as p
import newsrc.model as m

class Tuning:

    def __init__(self):
        self.input_args = self.load_input_args()
        self.nominationPopulation = p.PeerNominatedDataPopulation('Peer-Nominated data population', self.input_args, 'yes')
        self.communicationPopulation = p.CommunicationDataPopulation('Communication data population', self.input_args, 'yes')
        self.model = m.DiffusionModel('Diffusion Model', self.input_args)


    def load_input_args(self):
        try:
            input_args = json.loads(open('../input/tuning.json').read())
        except Exception as ex:
            print('tuning.json does not exist!')
            print(ex)

        return input_args

    def simulate(self, pop, thres, ipa, time):

        # outcomes of the intervention
        simulation_outcomes_child = {}
        simulation_outcomes_avg = {}

        for classroom_population in pop.get_class_graphs(pop.graph):
            classroom_population_id = list(classroom_population.nodes(data='class'))[1][1]
            simulation_outcomes_child[str(classroom_population_id)] = {}
            simulation_outcomes_avg[str(classroom_population_id)] = {}

            cl_pop = classroom_population

            # set parameters
            self.model.setThresholdPA(thres)
            self.model.setIPA(ipa)

            # running the simulation, executing the model with every timestamp
            for t in range(0, time):
                cl_pop = self.model.execute(cl_pop, t)

            outcomes_in_dict = self.get_PA_dictionary(cl_pop)
            simulation_outcomes_child[str(classroom_population_id)] = outcomes_in_dict
            simulation_outcomes_avg[str(classroom_population_id)] = outcomes_in_dict.mean(axis=1)

        return simulation_outcomes_child, simulation_outcomes_avg

    def execute(self, thres, ipa, t, population_name):
        '''
        Perform a grid search:
        1. define parameter space: ranging eahc parameter from 0 to 1 with steps of 0.05 (bins=20)
        2. run the model for each parameter combination
        3. assesses the goodness-of-fit
        '''
        generateGephiFiles = self.input_args['generateGephiFiles']
        writeToExcel = self.input_args['writeToExcel']


        # empirical data
        empirical_data = self.get_empirical_data(file = self.input_args['agent_pal_file'], classes = self.input_args['classes'])

        # population
        if (population_name == 'peer'):
            population = self.nominationPopulation
        elif (population_name == 'communication'):
            population = self.communicationPopulation

        list_error = []
        list_child = []
        list_cl = []

        # Run the model
        init_time = time()
        sim_child, sim_cl = self.simulate(population, thres, ipa, t)
        end_time = time()

        # Goodness-of-fit
        new_gof = self.get_error(graph=sim_cl, empirical=empirical_data)
        list_error = ((thres, ipa, new_gof))
        list_child = (sim_child)
        list_cl = (sim_cl)

        # Print progress
        print('thres_PA:', thres, ' I_PA:', ipa, ' error:', new_gof, '|runtime:', (end_time - init_time))

        return list_error, list_child, list_cl, empirical_data

    def execute_grid_search(self, t_range, i_range, t, population_name):
        '''
        Perform a grid search:
        1. define parameter space: ranging eahc parameter from 0 to 1 with steps of 0.05 (bins=20)
        2. run the model for each parameter combination
        3. assesses the goodness-of-fit
        '''
        generateGephiFiles = self.input_args['generateGephiFiles']
        writeToExcel = self.input_args['writeToExcel']

        thres_mesh = t_range
        I_PA_mesh = i_range

        # empirical data
        empirical_data = self.get_empirical_data(file = self.input_args['agent_pal_file'], classes = self.input_args['classes'])

        # population
        if (population_name == 'peer'):
            population = self.nominationPopulation
        elif (population_name == 'communication'):
            population = self.communicationPopulation

        list_error = []
        list_child = []
        list_cl = []
        for thres in thres_mesh:
            for i in I_PA_mesh:
                # Run the model
                init_time = time()
                sim_child, sim_cl = self.simulate(population, thres, i, t)
                end_time = time()

                # Goodness-of-fit
                new_gof = self.get_error(graph=sim_cl, empirical=empirical_data)
                list_error.append((thres, i, new_gof))
                list_child.append(sim_child)
                list_cl.append(sim_cl)

                # Print progress
                print('thres_PA:', thres, ' I_PA:', i, ' error:', new_gof, '|runtime:', (end_time - init_time))

        return list_error, list_child, list_cl, empirical_data


    def get_error(self, graph, empirical):
        '''
        Calculates the sum of squared errors (SSE)
        '''

        sim = pd.DataFrame(graph)
        sim = sim.iloc[[0, 364, 699],].T
        sim.columns = ['W1', 'W4', 'W5']

        empirical.columns = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']
        empirical = empirical.set_index(sim.index)

        # SSE wave 4
        #error_w4 = ((sim[['W4']] - empirical[['W4']]) ** 2).sum().sum()

        # SSE wave 5
        error_w5 = ((sim[['W5']] - empirical[['W5']]) ** 2).sum().sum()

        # Divided by 10 to increase the chance of acceptance of worst scenarios
        # return ((PA_sim - empirical)**2).sum().sum()/10, parameters
        return error_w5


    def get_empirical_data(self, file, classes):
        '''
        Get empirical physical activity data.

        Args:
            metric (str): physical activity metrics to use. default is number of steps.
            classes (array): list of class ids

        Returns:
            dataframe: physical activity data (steps) per child and wave.
        '''

        df_pal = pd.read_csv(file, sep=';', header=0, encoding='latin-1')
        df_pal = df_pal[df_pal['Class'].isin(classes)]

        df_pal = df_pal.groupby(['Class', 'Wave']).mean()['Steps'].reset_index()
        df_pal.Steps = df_pal.Steps * 0.000153
        df_pal = df_pal.pivot(index='Class', columns='Wave')['Steps']

        return df_pal

    def get_PA_dictionary(self, graph):
        results_dict = dict(graph.nodes(data=True))
        PA_dict = {}
        for k, v in results_dict.items():
            PA_dict[k] = results_dict[k]['PA_hist']

        return pd.DataFrame(PA_dict)

    def execute_simulated_annealing(self, initial_parameters, t, population_name):
        '''
        Parameter tuning function using simulated annealing
        '''

        # Get empirical data
        empirical_data = self.get_empirical_data(file=self.input_args['agent_pal_file'], classes=self.input_args['classes'])

        # Get population data
        if (population_name == 'peer'):
            population = self.nominationPopulation
        elif (population_name == 'communication'):
            population = self.communicationPopulation

        # Keeping history (vectors)
        error_hist = list([])
        parameters_hist = list([])

        # run with initial parameters
        init_time = time()
        sim_child, sim_cl = self.simulate(population, initial_parameters[0], initial_parameters[1], t)
        end_time = time()

        # Goodness-of-fit of initial parameters
        old_error = self.get_error(graph=sim_cl, empirical=empirical_data)

        # Save simulation and parameters
        error_hist.append(old_error)
        parameters_hist.append(initial_parameters)

        # Print progress
        print('thres_PA: ', initial_parameters[0], " I_PA: ", initial_parameters[1], "|runtime: ", (end_time - init_time))

        # Simulated annealing settings
        T = 1.0 # initial temperature
        T_min = 0.01 # minimum temperature
        alpha = 0.9 # cooling factor
        n_neighbors = 20 # number of neighbors explored

        parameters = initial_parameters

        while T > T_min:
            print('\nTemp: ', T)
            i = 1
            while i <= n_neighbors:
                init_time = time()
                new_parameters = self.get_neighbor(parameters)
                # run with new parameters
                sim_child, sim_cl = self.simulate(population, new_parameters[0], new_parameters[1], t)

                # Goodness-of-fit of new parameters
                new_error = self.get_error(graph=sim_cl, empirical=empirical_data)
                end_time = time()

                # Print progress
                print(T, i, 'thres_PA: ', new_parameters[0], " I_PA: ", new_parameters[1], 'cost: ', new_error,
                      '|runtime: ',(end_time - init_time))

                if new_error < old_error:
                    parameters = new_parameters
                    parameters_hist.append(parameters)
                    old_error = new_error
                    error_hist.append(old_error)
                else:
                    ap = self.get_acceptance_probability(old_error, new_error, T)
                    if ap > random.random():
                        # print 'accepted!'
                        parameters = new_parameters
                        parameters_hist.append(parameters)
                        old_error = new_error
                        error_hist.append(old_error)
                i += 1
            pprint(parameters_hist[-1])
            print(error_hist[-1])
            T = T * alpha

        return parameters, error_hist, parameters_hist

    def get_neighbor(self, parameters):
        '''

        Parameters are:
            thres_PA = 0.2
            I_PA = 0.00075
        A list with two positions:
            [thres, I_PA]
        '''
        old_thres = parameters[0]
        old_I_PA = parameters[1]

        minn = 0.00001

        max_I_PA = 0.05
        inf_I_PA = -0.005
        sup_I_PA = 0.005

        maxn_thres = 0.9999
        inf_thres = -0.1
        sup_thres = 0.1

        new_thres = old_thres + ((sup_thres - inf_thres) * random.random() + inf_thres)
        new_thres = minn if new_thres < minn else maxn_thres if new_thres > maxn_thres else new_thres

        new_I_PA = old_I_PA + ((sup_I_PA - inf_I_PA) * random.random() + inf_I_PA)
        new_I_PA = minn if new_I_PA < minn else max_I_PA if new_I_PA > max_I_PA else new_I_PA

        return [new_thres, new_I_PA]

    def get_acceptance_probability(self, old_error, new_error, T):
        '''
        Function to define acceptance probability values for Simulated Annealing
        Args:
            old_error: goodness-of-fit of old parameters
            new_error: goodness-of-fit of new parameters
            T: temperature of simulated annealing
        Returns: acceptance probability
        '''

        delta = new_error - old_error
        probability = np.exp(-delta / T)

        return probability



    def resultsToExcel(self, results):
        # loop the classes
        for class_id, res in results.items():

            directory = '../output/Class' + repr(int(float(class_id)))

            if not os.path.exists(directory):
                os.makedirs(directory)

            filename = directory + '/' + repr(int(float(class_id))) + '.xls'

            w = Workbook()
            ws = w.add_sheet(class_id.strip('\'') + ' Simulation Outcomes')

            # loop the dataframe
            rowNum = 0
            colNum = 0

            for i in self.input_args['intervention_strategy']:
                col = list(res[i])
                colLen = len(col)

                # writing the labels in excel sheet
                ws.write(rowNum, colNum, i)
                for c in col:
                    rowNum = rowNum + 1
                    ws.write(rowNum, colNum, c)

                colNum = colNum + 1
                rowNum = 0

            w.save(filename)


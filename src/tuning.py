import numpy as np
import pandas as pd
import random
from pprint import pprint
from time import time

import src.population as p
import src.model as m
import src.utils as utils

class Tuning:
    '''
        Tuning class. Algorthims for parameter tuning.
    '''

    def __init__(self):
        self.input_args = utils.load_input_args('../input/simulation.json')
        self.nominationPopulation = p.PeerNominatedDataPopulation('Peer-Nominated data population', self.input_args)
        self.communicationPopulation = p.CommunicationDataPopulation('Communication data population', self.input_args)
        self.model = m.DiffusionModel(self.input_args)

    def execute(self):
        pass

    def simulate(self, pop, thres, ipa, time):
        '''
        Performs a single simulation

        Args:
            pop (Graph): agent population
            thres (float): threshold value
            ipa (float): I_PA value
            time (integer): duration of simulation in days

        Returns:
            simulated output per child in a classroom and the average per classroom
        '''

        # set parameters
        self.model.setThresholdPA(thres)
        self.model.setIPA(ipa)

        # outcomes of the intervention
        simulation_outcomes_child = {}
        simulation_outcomes_class = {}

        # for each classroom
        for class_population in pop.get_class_graphs(pop.graph):
            class_id = list(class_population.nodes(data='class'))[1][1]
            simulation_outcomes_child[str(class_id)] = {}
            simulation_outcomes_class[str(class_id)] = {}

            # agents in classroom
            cl_pop = class_population.copy()

            # run the simulation for each day (t)
            for t in range(0, time):
                sim_pop = self.model.execute(cl_pop, t)

            # convert output in a dictionary
            outcomes_in_dict = utils.get_PA_dictionary(sim_pop)
            # add output in outcome objects
            simulation_outcomes_child[str(class_id)] = outcomes_in_dict
            simulation_outcomes_class[str(class_id)] = outcomes_in_dict.mean(axis=1)

        return simulation_outcomes_child, simulation_outcomes_class

    def get_error(self, graph, empirical, time):
        '''
        Calculates the error between simulated and observed data

        Args:
            graph (Graph): model output
            empirical (dictionary): empirical data

        Returns:
            sum of squared errors (SSE)
        '''

        # extract model output at baseline (W1), 1 year (W4), and 2 years (W5)
        sim = pd.DataFrame(graph)
        sim = sim.iloc[[0, (time-1)]].T
        sim.columns = ['W1', 'W5']

        # rename columns of empirical data
        empirical.columns = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7']
        empirical = empirical.set_index(sim.index)

        # SSE wave 4 (not used)
        #error_w4 = ((sim[['W4']] - empirical[['W4']]) ** 2).sum().sum()

        # calculate SSE of W5
        error_w5 = (((sim[['W5']]) - (empirical[['W5']])) ** 2).sum().sum()

        #print(empirical[['W5']])

        return error_w5

class GridSearch(Tuning):
    '''
        Grid search subclass.
    '''
    def __init__(self):
        super(GridSearch, self).__init__()

    def execute(self, t_range, i_range, t, population_name):
        '''
        Performs a grid search: (1) run the model for each parameter combination (2) calculate the goodness-of-fit

        Args:
            t_range (array): threshold values
            i_range (array): i_pa values
            t (integer): simulation time in days
            population_name (string): network selection, i.e. nomination or communication

        Returns:
            SSE, output per child by classrom, output per classroom, and empirical data for each parameter combination
        '''

        # empirical data
        empirical_data = utils.get_empirical_data(file = self.input_args['agent_pal_file'], classes = self.input_args['classes'])

        # population
        if (population_name == 'nomination'):
            population = self.nominationPopulation
        elif (population_name == 'communication'):
            population = self.communicationPopulation

        list_error = []
        list_child = []
        list_class = []
        # Run the model for each parameter combination of thres_mesh and i_PA_mesh
        for thres in t_range:
            for i in i_range:
                # Run the model
                init_time = time()
                sim_child, sim_cl = self.simulate(population, thres, i, t)
                end_time = time()

                # Goodness-of-fit
                new_gof = self.get_error(graph=sim_cl, empirical=empirical_data, time=t)
                list_error.append((thres, i, new_gof))
                list_child.append(sim_child)
                list_class.append(sim_cl)

                # Print progress
                print('thres_PA:', thres, 'I_PA:', i, 'error:', new_gof, 'runtime:', (end_time - init_time))

        return list_error, list_child, list_class, empirical_data

    def executeSet(self, param_set, t, population_name):
        '''
        Run the model for a single parameter combination

        Args:
            param_set: set of parameter combinations
            t (integer): simulation time in days
            population_name (string): network selection, i.e. nomination or communication

        Returns:
            SSE, output per child by classrom, output per classroom, and empirical data
        '''

        # empirical data
        empirical_data = utils.get_empirical_data(file=self.input_args['agent_pal_file'],
                                                 classes=self.input_args['classes'])

        # population
        if (population_name == 'nomination'):
            population = self.nominationPopulation
        elif (population_name == 'communication'):
            population = self.communicationPopulation

        list_error = []
        list_child = []
        list_class = []
        for params in param_set:
            thres = params[0]
            ipa = params[1]
            # Run the model
            init_time = time()
            sim_child, sim_cl = self.simulate(population, thres, ipa, t)
            end_time = time()

            # Goodness-of-fit
            new_gof = self.get_error(graph=sim_cl, empirical=empirical_data, time = t)
            list_error.append((thres, ipa, new_gof))
            list_child.append(sim_child)
            list_class.append(sim_cl)

            # Print progress
            print('thres_PA:', thres, 'I_PA:', ipa, 'error:', new_gof, 'runtime:', (end_time - init_time))

        return list_error, list_child, list_class, empirical_data


class SimulatedAnnealing(Tuning):
    '''
       Simulated annealling subclass.
    '''
    def __init__(self):
        super(SimulatedAnnealing, self).__init__()

    def execute(self, t_initial, i_initial, t, population_name):
        '''
        Performs simulated annealing for parameter tuning. Finds an optimal solution

        Args:
            t_initial (float): initial threshold value
            i_initial (float): initial i_pa value
            t (integer): simulation time in days
            population_name (string): network selection, i.e. nomination or communication

        Returns:
            optimal parameters, and a list of errors and associated parameter values
        '''

        initial_parameters = [t_initial, i_initial]

        # Get empirical data
        empirical_data = self.get_empirical_data(file=self.input_args['agent_pal_file'], classes=self.input_args['classes'])

        # Get population data
        if (population_name == 'nomination'):
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
                print(T, i, 'thres_PA:', new_parameters[0], "I_PA:", new_parameters[1], 'SSE:', new_error,
                      'runtime:',(end_time - init_time))

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
            Gets neighbor for simualted annealing algorithm

            Args:
                parameters (list): [threshold, I_PA]

            Returns:
                new parameter combination
        '''

        old_thres = parameters[0]
        old_I_PA = parameters[1]

        minn = 0.00001

        max_I_PA = 0.4
        inf_I_PA = -0.05
        sup_I_PA = 0.05

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
            Function to define acceptance probability values for Simulated Annealing.

            Args:
                old_error (float): goodness-of-fit of old parameters
                new_error (float): goodness-of-fit of new parameters
                T (float): temperature of simulated annealing

            Returns:
                acceptance probability
        '''

        delta = new_error - old_error
        probability = np.exp(-delta / T)

        return probability



from networkPA import generate_network_PA,get_empirical
from models import contagion_model
from classes_analysis import get_class_graphs

import networkx as nx
import numpy as np
import pandas as pd
from pprint import pprint
from random import random
from time import time

def grid_search(bins=10, label='all', level_f=''):
    '''
    Seek the information a grid divided by the number of bins given
    '''
    thres_mesh = np.arange(0.0, 0.1,1/bins)
    empirical_data = get_empirical('')
    original_graph = generate_network_PA('', label=label)
    classes_graph=get_class_graphs(original_graph.copy(), label=label)
        
    list_solutions = []
    
    for t in thres_mesh:
        print('Thresh: ', t)
        for gr in classes_graph:
            new_cost, _ = get_error(graph=gr.copy(), empirical=empirical_data, parameters=[t], label=label)
            list_solutions.append((t,new_cost,gr.graph['class']))
                
    return list_solutions


def get_error(graph, empirical, parameters=None, label='all'):
    '''
    Runs the simulation and calculates the difference
    '''

    if parameters is None:
        delta = random()
    else:
        delta = parameters[0]
    
    #graph = generate_network_PA(level_f=level_f, label=label)
    #print('PA hist <before>: ', nx.get_node_attributes(graph, 'PA_hist'))
    
    init_time = time()
    contagion_model(graph, years=1, delta=delta)
    end_time = time()
    print('Social Contagion Running Time: ', (end_time-init_time))
    #print(parameters, '\n')

    #print('PA hist <after>: ', nx.get_node_attributes(graph, 'PA_hist'))
    PA_results = {}
    for node in graph.nodes():
        PA_results[node] = graph.nodes[node]['PA_hist']

    PA_df = pd.DataFrame(PA_results).T

    PA_sim = PA_df[[0, 30, 60, 364]]
    PA_sim.columns = ['W1', 'W2', 'W3', 'W4']
    empirical.columns = ['W1', 'W2', 'W3', 'W4']
    print('PA_sim')
    print(PA_sim)
    print('empirical')
    print(empirical)
    # Changes to penalize W4 more than the others.
    
    # For v1
    #error = ((PA_sim[['W1', 'W2', 'W3']]-empirical[['W1', 'W2', 'W3']])**2).sum().sum() + (((PA_sim.W4 - empirical.W4)*4)**2).sum().sum()

    # For v2
    error = ((PA_sim[['W1', 'W4']]-empirical[['W1', 'W4']])**2).sum().sum()

    # Divided by 100 to increase the chance of acceptance of worst scenarios
    # return ((PA_sim - empirical)**2).sum().sum()/100, parameters
    return error/10, parameters


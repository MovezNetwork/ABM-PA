'''
Methods related to graph creation using NetworkX library. The graph represents the school class social networks.
'''


import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import random

def generate_network_PA(level_f='../', label=None, formula_s=None, debug=False):
    '''
    Calls methods for creation of the graph and saves the graph as gexf file.
    
    Args:
        level_f (str): filesystem level
        label (str): label of the graph- gen, all or friends graph
        formula_s (str): string formula to customize the calculation of the edges
        debug (boolean): debug related messages. default is false

    Returns:
    Graph: NetworkX graph representing school classes network.
    
    '''
    print('###############################################################')
    print('Graph generation starting!')
    print('Label: {}\nFormula: {}'.format(label, formula_s))
    print('###############################################################\n')
    graph = nx.DiGraph()
    if debug:
        print('Create connections...')
    create_connections(graph=graph, level_f=level_f, label=label, formula_s=formula_s, waves='all')
    if debug:
        print('Nodes after connections: #', len(graph.nodes()))
        print('Edges created #: ', len(graph.edges()))
        print('\nCreate agents...')
    create_agents_PA(graph=graph, level_f=level_f)

    # Comment this if you want to keep all the nodes
    if debug:
        print('Removing nodes not in the specified classes...')
    remove_nodes_PA(graph=graph, level_f=level_f)
    if debug:
        print('Nodes remaining after removal: #', len(graph.nodes()))
        print('Edges remaining after removal #: ', len(graph.edges()))
    if label is None:
        g_file = "../output/graph.gexf"
    else:
        g_file = 'graph_' + label + '.gexf'
    try:
        nx.write_gexf(graph, g_file)
    except IOError as e:
        errno, strerror = e.args
        print("I/O error({0}): {1}".format(errno,strerror))
        # e can be printed directly without using .args:
        # print(e)
    
    print('###############################################################')
    print('Graph generated successfuly!')
    print('###############################################################\n')
    return graph

def create_agents_PA(graph, level_f='../'):
    
    '''
    Calls methods for customizing the graph nodes, with information like gender, age, class, height, weight, environment, BMI. 
    
    Args:
        level_f (str): filesystem level
        graph (Graph): The input school graph

    Returns:
    Graph: Updated graph with customized nodes.
    
    '''
    
    PA_dict = generate_PA(metric='steps', level_f=level_f)
    gender_dict, age_dict, class_dict = generate_basic(level_f=level_f)
    environment_dict = generate_environment(level_f=level_f)
    bmi_dict = generate_bmi(level_f=level_f)
    
    PA_dict = fix_float64(PA_dict)
    #print('PA')
    gender_dict = fix_float64(gender_dict)
    #print('gender')
    age_dict = fix_float64(age_dict)
    #print('age')
    class_dict = fix_float64(class_dict)
    #print('class')
    environment_dict = fix_float64(environment_dict)
    #print('env')
    bmi_dict = fix_float64(bmi_dict)
    #print('obesity classifier')
    
    nx.set_node_attributes(graph, values=PA_dict, name='PA')
    nx.set_node_attributes(graph, values=gender_dict, name='gender')
    nx.set_node_attributes(graph, values=age_dict, name='age')
    nx.set_node_attributes(graph, values=class_dict, name='class')
    nx.set_node_attributes(graph, values=environment_dict, name='env')
    nx.set_node_attributes(graph, values=bmi_dict, name='bmi')
    
    # Adding category for the nodes
    obesity_class = {}
    for node in graph.nodes():
        obesity_class[node] = get_bmi_cat(gender_dict[node], age_dict[node], bmi_dict[node])

    nx.set_node_attributes(graph, values=obesity_class, name='bmi_cat')

    return graph

def remove_nodes_PA(graph, level_f='../'):
    
    '''
    Remove the nodes that are not part of the classes of interest.
    
    Args:
        level_f (str): filesystem level
        graph (Graph): The input school graph

    Returns:
    Graph: Updated graph with (potentially) removed nodes.
    
    '''
        

    nodes_removed_class = []
    
    try:
        input_simulation = json.loads(open('../input/simulation.json').read())
    except Exception as ex:
        print('simulation.json does not exist!')
        print(ex)
        return
    
    
    class_list = input_simulation['classes']
    
    for node in graph.nodes():
        if graph.nodes()[node]['class'] not in class_list:
            nodes_removed_class.append(node)

    graph.remove_nodes_from(nodes_removed_class)
    

    print('Nodes removed for not being in the selected classes: #', len(nodes_removed_class))
    return graph



def generate_PA(metric='steps', level_f='../'):
    
    
    
    '''
    Generate physical activity value for nodes.
    
    Args:
        level_f (str): filesystem level
        metric (str): physical activity metrics to use. default is number of steps.

    Returns:
    dictionary: Dictionary with average steps per child and per wave.
    
    '''
    

    fitbit = pd.read_csv('../data/fitbit.csv', sep=';', header=0)
    
    steps_mean_wave = fitbit.groupby(['Child_Bosse', 'Wave']).mean()['Steps_ML_imp1'].reset_index()
    steps_mean_wave.Steps_ML_imp1 = steps_mean_wave.Steps_ML_imp1 * 0.000153
    steps_mean_wave = steps_mean_wave.pivot(index='Child_Bosse', columns='Wave')['Steps_ML_imp1']

    return dict(steps_mean_wave[1])



    
    '''
    NetworkClass:       Does the class reach the treshold of >60% of participation
    Steps:              observed mean daily steps count per week    
    Minutes_MVPA: :     observed mean daily minutes of moderate to vigorous physical activity per week      
    Steps_imp1:         simple imputation for missing Steps data    
    MVPA_imp1:          simple imputation for missing Minutes_MVPA data 
    Steps_ML_imp1:      single multilevel imputation for Steps data 
    Minutes_MVPA_ML_imp1:   single multilevel imputation for Minutes_MVPA data

    |- inputs:
        |-- metric: steps or mvpa
    |- outputs:
        |-- dictionary with steps or minutes

    '''
    
    '''
    # Mean of minutes from moderate to vigorous activity and steps (all imputed)
    minutes_MVPA_df = fitbit.groupby(['Child_Bosse']).mean()['Minutes_MVPA_ML_imp1']

    # Steps are converted to fit the system. 1.53 (PA) corresponds to 10.000 steps.
    steps_df = fitbit.groupby(['Child_Bosse']).mean()['Steps_ML_imp1'] * 0.000153

    if metric == 'mvpa':
        minutes_MVPA_df.to_csv(level_f+'output/mvpa.csv')
        return dict(minutes_MVPA_df)
    elif metric == 'steps':
        steps_df.to_csv(level_f+'output/steps.csv')
        return dict(steps_df)
    else:
        print('Metric not valid! >>>', metric)
        return False

    '''


def generate_environment(level_f='../'):
    
    '''
    Generate environment value for nodes. Combination of different questionnaire responses for owning computers, car, ownroom or allowing summer vacation.
    
    Args:
        level_f (str): filesystem level

    Returns:
    dictionary: Dictionary with environment score per child.
    
    '''

    env = pd.read_csv('../data/environment.csv', sep=';', header=0)
    env = env[['Child_Bosse', 'School', 'Class', 'Wave', 'GEN_FAS_computer_R',
               'GEN_FAS_car_R', 'GEN_FAS_vacation_R', 'GEN_FAS_ownroom_R']]
    
    try:
        input_simulation = json.loads(open('../input/simulation.json').read())
    except Exception as ex:
        print('simulation.json does not exist!')
        print(ex)
        return
    
    
    classes = input_simulation['classes']
    
    env = env[env['Class'].isin(classes)]

    env_filter = env[env.Wave==1][['Child_Bosse', 'GEN_FAS_computer_R', 'GEN_FAS_car_R', 
                                   'GEN_FAS_vacation_R', 'GEN_FAS_ownroom_R']].copy()
    
    env_filter['FAS_Score_R'] = env_filter['GEN_FAS_computer_R'] + env_filter['GEN_FAS_vacation_R'] + \
                            env_filter['GEN_FAS_car_R']*1.5 + env_filter['GEN_FAS_ownroom_R']*3

    # To keep the values between 0 and 2.
    env_filter.FAS_Score_R = env_filter.FAS_Score_R/6
    env_filter.index = env_filter['Child_Bosse']

    env_dict = dict(env_filter['FAS_Score_R'])
    for key, value in env_dict.items():
        if np.isnan(value):
            env_dict[key] = env_filter.FAS_Score_R.mean()

    return env_dict


def generate_bmi(level_f='../'):
    
        
    '''
    Generate BMI value for nodes.
    
    Args:
        level_f (str): filesystem level

    Returns:
    dictionary: Dictionary with BMI score per child.
    
    '''

    bmi = pd.read_csv('../data/bmi.csv', sep=';', header=0)
    bmi = bmi[bmi.Wave==2][['Child_Bosse', 'BMI']]
    bmi.index = bmi.Child_Bosse
    bmi = bmi['BMI']

    return dict(bmi)

def fix_float64(orig_dict):
    
    '''
    Helper method converts the numpy.float64 values from a dictionary to native float type.
    
    Args:
        level_f (str): filesystem level

    Returns:
    dictionary: Dictionary with updated float values. 
    '''
    

    new_dict = {}
    for k, item in orig_dict.items():
        try:    
            new_dict[k] = -1.0 if np.isnan(item) else np.asscalar(item)
        except:
            new_dict[k] = -1.0
            #print(k, item)
    return new_dict

def generate_basic(level_f='../'):
        
    '''
    Generate gender, age and class information per node.
    
    Args:
        level_f (str): filesystem level

    Returns:
    dictionary: Dictionary with gender, age and class information per child.
    
    '''
    background = pd.read_csv('../data/background.csv', sep=';', header=0)
    pp = pd.read_csv('../data/pp.csv', sep=';', header=0)

    gender_df = background.groupby(['Child_Bosse']).mean()['Gender']
    age_df = background.groupby(['Child_Bosse']).mean()['Age']
    
    # Generate Class
    pp['class'] = pp.Class_Y1
    # Fill the missing data at Class column with the data from Y1.
    pp['class'].fillna(pp.Class_Y2, inplace=True)
    pp.index = pp.Child_Bosse
    class_df = pp['class']

    
    gender_df.to_csv('../output/gender.csv')
    age_df.to_csv('../output/age.csv')
    class_df.to_csv('../output/class.csv')
    
    return dict(gender_df), dict(age_df), dict(class_df)

def get_empirical(metric='steps', level_f='../',classes=[]):
    '''
    Get empirical physical activity data. 
    
    Args:
        level_f (str): filesystem level
        metric (str): physical activity metrics to use. default is number of steps.
        classes (array): list of class ids

    Returns:
    dataframe: physical activity data (steps) per child and wave.
    
    '''
    fitbit = pd.read_csv('../data/fitbit.csv', sep=';', header=0)
    
    try:
        input_simulation = json.loads(open('../input/simulation.json').read())
    except Exception as ex:
        print('simulation.json does not exist!')
        print(ex)
        return
    
    
    classes = input_simulation['classes']
    
    fitbit = fitbit[fitbit['Class'].isin(classes)]
    steps_mean_wave = fitbit.groupby(['Child_Bosse', 'Wave']).mean()['Steps_ML_imp1'].reset_index()
    steps_mean_wave.Steps_ML_imp1 = steps_mean_wave.Steps_ML_imp1 * 0.000153
    steps_mean_wave = steps_mean_wave.pivot(index='Child_Bosse', columns='Wave')['Steps_ML_imp1']

    return steps_mean_wave


def create_connections(graph, formula_s=None, label=None, waves='all', level_f='../'):
    
    '''
    Create graph's weights between edges depending on the chosen formula. 
    The connections are created and weighted based on questionnaire responses. 
    Use all (the whole set of questions), gen (only the general questions) or friends (only the friends questions) to create the weights
    
    Args:
        graph (Graph): input graph
        level_f (str): filesystem level
        label (str): label of the graph- gen, all or friends graph
        formula_s (str): string formula to customize the calculation of the edges
        waves (str): waves of data to include. 

    Returns:
    Graph: updated graph with weighted connections.
    
    '''
        

    # List with all the participants in the experiment
    pp = pd.read_csv('../data/pp.csv', sep=';', header=0)
    list_participants = list(pp.Child_Bosse)

    # Read the file with the nominations from the participants and adapt the labels for the columns
    nominations = pd.read_csv('../data/nominations.csv', sep=';', header=0)
    # nominations.columns = ['class', 'child', 'wave', 'variable', 'class_nominated', 'nominated', 'same_class']

    # Read formula to calculate the weight for the connections
    if formula_s is None:
        try:
            if label is None:
                formula = json.loads(open('../input/connections.json'.format(level_f)).read())
            elif label=='all':
                formula = json.loads(open('../input/connections_all.json'.format(level_f, label)).read())
            elif label=='gen':
                formula = json.loads(open('../input/connections_gen.json'.format(level_f, label)).read())
            elif label=='friend':
                formula = json.loads(open('../input/connections_friend.json'.format(level_f, label)).read())
        except Exception as ex:
            print('File {}settings/connections_{}.json does not exist!'.format(level_f, label))
            print(ex)
            return
    else:
        try:
            formula = json.loads(formula_s)
        except:
            print('Formula provided is corrupted.')
            return
    

    # Sum of all weights from the formula
    max_score = sum(formula.values())

    # Create a dictionary with the connections and weights
    connections_dict = {}
    for child in list(pp.Child_Bosse):
        connections_dict[child] = {}

    # To avoid repetition of nominations in different waves
    nominations_list = []

    for line in nominations[['child', 'nominated', 'variable']].iterrows():
        (ch, nom, var) = line[1]  
        # Verify if nominated is in the list of participants (pp)
        if nom in list_participants and (ch, nom, var) not in nominations_list:
            # Add value in the key
            connections_dict[ch][nom] = connections_dict[ch].get(nom, 0) + 1*formula[var]
            nominations_list.append((ch, nom, var))
    
    # Make a dataframe and normalize the values for the edges
    connections_df = pd.DataFrame(connections_dict).fillna(0)/max_score
    connections_dict = connections_df.to_dict()
    # eric approach : child(node[0])-succ;nominated(node[1])-pred;weight
    # thabo approach: child(node[0])-pred;nominated(node[1])-succ;weight
    
    #An arrow (x, y) is considered to be directed from x to y; y is called the head and x is called the tail of the arrow; y is said to be a direct successor of x and x is said to be a direct predecessor of y.
    
    # Create the edges in the graph
    for node in connections_dict.items():
        pred = node[0]
        origins = node[1]
        for succ, weight in origins.items():
            if weight > 0:
                graph.add_edge(pred,succ,weight=weight)
#                 print('pred: '+ repr(pred)+' succ:'+repr(succ)+' weight:'+repr(weight))

    # Save the connections file in the results folder
    
    if label is None:
        connections_df.to_csv('../output/connections.csv')
    else:
        connections_df.to_csv(('../output/connections_{1}.csv').format(level_f, label))

    return graph

if __name__ == "__main__":
    # execute only if run as a script
    graph = generate_network()

def get_bmi_cat(gender,age,bmi):
    
    '''
    Calculating the BMI category based on gender, age and BMI value 
    
    Args:
        gender (Integer): person gender 
        age (Integer): person age 
        bmi (Integer): person BMI 

    Returns:
    Integer: BMI category value
    
    '''


    if (bmi == -1) or (gender == -1) or (age == -1) :
        return np.nan

    category=0
    #males
    if gender==0:
        if age==2:
            if bmi<=13.36:
                category=1
            elif 13.37<=bmi<=15.13:
                category=2
            elif 15.14<=bmi<=18.40:
                category=3
            elif 18.41<=bmi<=20.09:
                category=4
            elif bmi>20.09:
                category=5
        elif age==3:
            if bmi<=13.09:
                category=1
            elif 13.10<=bmi<=14.73:
                category=2
            elif 14.74<=bmi<=17.88:
                category=3
            elif 17.89<=bmi<=19.57:
                category=4
            elif bmi>19.57:
                category=5
        elif age==4:
            if bmi<=12.86:
                category=1
            elif 12.87<=bmi<=14.42:
                category=2
            elif 14.43<=bmi<=17.54:
                category=3
            elif 17.55<=bmi<=19.29:
                category=4
            elif bmi>19.29:
                category=5
        elif age==5:
            if bmi<=12.66:
                category=1
            elif 12.67<=bmi<=14.20:
                category=2
            elif 14.21<=bmi<=17.41:
                category=3
            elif 17.42<=bmi<=19.30:
                category=4
            elif bmi>19.30:
                category=5
        elif age==6:
            if bmi<=12.50:
                category=1
            elif 12.51<=bmi<=14.06:
                category=2
            elif 14.07<=bmi<=17.54:
                category=3
            elif 17.55<=bmi<=19.78:
                category=4
            elif bmi>19.78:
                category=5
        elif age==7:
            if bmi<=12.42:
                category=1
            elif 12.43<=bmi<=14.03:
                category=2
            elif 14.04<=bmi<=17.91:
                category=3
            elif 17.92<=bmi<=20.63:
                category=4
            elif bmi>20.63:
                category=5
        elif age==8:
            if bmi<=12.42:
                category=1
            elif 12.43<=bmi<=14.14:
                category=2
            elif 14.15<=bmi<=18.43:
                category=3
            elif 18.44<=bmi<=21.60:
                category=4
            elif bmi>21.60:
                category=5
        elif age==9:
            if bmi<=12.50:
                category=1
            elif 12.51<=bmi<=14.34:
                category=2
            elif 14.35<=bmi<=19.09:
                category=3
            elif 19.10<=bmi<=22.77:
                category=4
            elif bmi>22.77:
                category=5
        elif age==10:
            if bmi<=12.66:
                category=1
            elif 12.67<=bmi<=14.63:
                category=2
            elif 14.64<=bmi<=19.83:
                category=3
            elif 19.84<=bmi<=24.00:
                category=4
            elif bmi>24.00:
                category=5
        elif age==11:
            if bmi<=12.89:
                category=1
            elif 12.90<=bmi<=14.96:
                category=2
            elif 14.97<=bmi<=20.54:
                category=3
            elif 20.55<=bmi<=25.10:
                category=4
            elif bmi>25.10:
                category=5
        elif age==12:
            if bmi<=13.18:
                category=1
            elif 13.19<=bmi<=15.34:
                category=2
            elif 15.35<=bmi<=21.21:
                category=3
            elif 21.22<=bmi<=26.02:
                category=4
            elif bmi>26.02:
                category=5
        elif age==13:
            if bmi<=13.59:
                category=1
            elif 13.60<=bmi<=15.83:
                category=2
            elif 15.84<=bmi<=21.90:
                category=3
            elif 21.91<=bmi<=26.84:
                category=4
            elif bmi>26.84:
                category=5
        elif age==14:
            if bmi<=14.09:
                category=1
            elif 14.10<=bmi<=16.40:
                category=2
            elif 16.41<=bmi<=22.61:
                category=3
            elif 22.62<=bmi<=27.63:
                category=4
            elif bmi>27.63:
                category=5
        elif age==15:
            if bmi<=14.60:
                category=1
            elif 14.61<bmi<16.97:
                category=2
            elif 16.98<=bmi<=23.28:
                category=3
            elif 23.29<=bmi<=28.30:
                category=4
            elif bmi>28.30:
                category=5
        elif age==16:
            if bmi<=15.12:
                category=1
            elif 15.13<=bmi<=17.53:
                category=2
            elif 17.54<=bmi<=23.89:
                category=3
            elif 23.90<=bmi<=28.88:
                category=4
            elif bmi>28.88:
                category=5
        elif age==17:
            if bmi<=15.60:
                category=1
            elif 15.61<=bmi<=18.04:
                category=2
            elif 18.05<=bmi<=24.45:
                category=3
            elif 24.46<=bmi<=29.41:
                category=4
            elif bmi>29.41:
                category=5
        elif age==18:
            if bmi<=16.00:
                category=1
            elif 16.01<=bmi<=18.49:
                category=2
            elif 18.50<=bmi<=24.99:
                category=3
            elif 25.00<=bmi<=30.00:
                category=4
            elif bmi>30.00:
                category=5
                    #males
    if gender==1:
        if age==2:
            if bmi<=13.24:
                category=1
            elif 13.25<bmi<14.82:
                category=2
            elif 14.83<=bmi<=18.01:
                category=3
            elif 18.02<=bmi<=19.81:
                category=4
            elif bmi>19.81:
                category=5
        elif age==3:
            if bmi<=12.98:
                category=1
            elif 12.99<=bmi<=14.46:
                category=2
            elif 14.47<=bmi<=17.55:
                category=3
            elif 17.56<=bmi<=19.36:
                category=4
            elif bmi>19.36:
                category=5
        elif age==4:
            if bmi<=12.73:
                category=1
            elif 12.74<=bmi<=14.18:
                category=2
            elif 14.19<=bmi<=17.27:
                category=3
            elif 17.28<=bmi<=19.15:
                category=4
            elif bmi>19.15:
                category=5
        elif age==5:
            if bmi<=12.50:
                category=1
            elif 12.51<=bmi<=13.93:
                category=2
            elif 13.94<=bmi<=17.14:
                category=3
            elif 17.15<=bmi<=19.17:
                category=4
            elif bmi>19.17:
                category=5
        elif age==6:
            if bmi<=12.32:
                category=1
            elif 12.33<=bmi<=13.81:
                category=2
            elif 13.82<=bmi<=17.33:
                category=3
            elif 17.34<=bmi<=19.65:
                category=4
            elif bmi>19.65:
                category=5
        elif age==7:
            if bmi<=12.26:
                category=1
            elif 12.27<=bmi<=13.85:
                category=2
            elif 13.86<=bmi<=17.74:
                category=3
            elif 17.75<=bmi<=20.51:
                category=4
            elif bmi>20.51:
                category=5
        elif age==8:
            if bmi<=12.31:
                category=1
            elif 12.32<=bmi<=14.01:
                category=2
            elif 14.02<=bmi<=18.34:
                category=3
            elif 18.35<=bmi<=21.57:
                category=4
            elif bmi>21.57:
                category=5
        elif age==9:
            if bmi<=12.44:
                category=1
            elif 12.45<=bmi<=14.27:
                category=2
            elif 14.28<=bmi<=19.06:
                category=3
            elif 19.07<=bmi<=22.81:
                category=4
            elif bmi>22.81:
                category=5
        elif age==10:
            if bmi<=12.64:
                category=1
            elif 12.65<=bmi<=14.60:
                category=2
            elif 14.61<=bmi<=19.85:
                category=3
            elif 19.86<=bmi<=24.11:
                category=4
            elif bmi>24.11:
                category=5
        elif age==11:
            if bmi<=12.95:
                category=1
            elif 12.96<=bmi<=15.04:
                category=2
            elif 15.05<=bmi<=20.73:
                category=3
            elif 20.74<=bmi<=25.42:
                category=4
            elif bmi>25.42:
                category=5
        elif age==12:
            if bmi<=13.39:
                category=1
            elif 13.40<=bmi<=15.61:
                category=2
            elif 15.62<=bmi<=21.67:
                category=3
            elif 21.68<=bmi<=26.67:
                category=4
            elif bmi>26.67:
                category=5
        elif age==13:
            if bmi<=13.92:
                category=1
            elif 13.93<bmi<16.25:
                category=2
            elif 16.26<=bmi<=22.57:
                category=3
            elif 22.58<=bmi<=27.76:
                category=4
            elif bmi>27.76:
                category=5
        elif age==14:
            if bmi<=14.48:
                category=1
            elif 14.49<=bmi<=16.87:
                category=2
            elif 16.88<=bmi<=23.33:
                category=3
            elif 23.34<=bmi<=28.57:
                category=4
            elif bmi>28.57:
                category=5
        elif age==15:
            if bmi<=15.01:
                category=1
            elif 15.02<=bmi<=17.44:
                category=2
            elif 17.45<=bmi<=23.93:
                category=3
            elif 23.94<=bmi<=29.11:
                category=4
            elif bmi>29.11:
                category=5
        elif age==16:
            if bmi<=15.46:
                category=1
            elif 15.47<=bmi<=17.90:
                category=2
            elif 17.91<=bmi<=24.36:
                category=3
            elif 24.37<=bmi<=29.43:
                category=4
            elif bmi>29.43:
                category=5
        elif age==17:
            if bmi<=15.78:
                category=1
            elif 15.79<=bmi<=18.24:
                category=2
            elif 18.25<=bmi<=24.69:
                category=3
            elif 24.70<=bmi<=29.69:
                category=4
            elif bmi>29.69:
                category=5
        elif age==18:
            if bmi<=15.99:
                category=1
            elif 16.00<=bmi<=18.49:
                category=2
            elif 18.50<=bmi<=24.99:
                category=3
            elif 25.00<=bmi<=30.00:
                category=4
            elif bmi>30.00:
                category=5
    return category
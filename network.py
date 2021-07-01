import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import re
import random

def fix_float64(orig_dict):
    '''
    This function converts the numpy.float64 values from a dictionary to native float type.
    {k: np.asscalar(item) for k, item in orig_dict.items()}
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
    Static values. Age changes a little.
    For the class, we take Y2. In case the data is missing, we use Y1 from pp data frame.
    '''
    background = pd.read_csv('background.csv', sep=';', header=0)
    pp = pd.read_csv('pp.csv', sep=';', header=0)

    gender_df = background.groupby(['Child_Bosse']).mean()['Gender']
    age_df = background.groupby(['Child_Bosse']).mean()['Age']
    
    # Generate Class
    pp['class'] = pp.Class_Y1
    # Fill the missing data at Class column with the data from Y1.
    pp['class'].fillna(pp.Class_Y2, inplace=True)
    pp.index = pp.Child_Bosse
    class_df = pp['class']

    
    gender_df.to_csv('gender.csv')
    age_df.to_csv('age.csv')
    class_df.to_csv('class.csv')
    
    return dict(gender_df), dict(age_df), dict(class_df)


def create_connections(graph, formula_s=None, label=None, waves='all', level_f='../'):
    '''
    graph: DiGraph
    formula_s: string containing a json with the weights for each variable
    --------------------------------------------------------------
    Network connections are based on influence from the kids on each other. The variables used are:
    --------------------------------------------------------------
    Health
    --------------------------------------------------------------
    SOC_DI_Com_network: (1 item) with who participants talk about what they eat and drink
    SOC_DI_Impression_management: (1 item) who participants want to come across as somebody who eats and drinks healthy
    SOC_Di_Modelling_reversed: (1 item) who are examples in eating & drinking healthy
    SOC_DI_Modelling: (1 item) who are eating & drinking products participants also want to eat or drink
    --------------------------------------------------------------
    Leadership and influence
    --------------------------------------------------------------
    SOC_GEN_Advice: (1 item) to who participants go to for advice
    SOC_GEN_Friendship: (1 item) with who participants are friends
    SOC_GEN_INNOV: (1 item) who most often have the newest products & clothes
    SOC_GEN_Leader: (1 item) who participants consider as leaders 
    SOC_GEN_Respect: (1 item) who participants respect
    SOC_GEN_Social_Facilitation: (1 item) with who participants hang out / have contact with
    SOC_GEN_Want2B: (1 item) who participants want to be like
    SOC_ME_Com_network: (1 item) with who participants talk about what they see on television  or internet 
    SOC_PA_Com_network: (1 item) with who participants talk about physical activity and sports
    SOC_PA_Impression_management: (1 item) for who participants want to come across as somebody who is doing sports often
    SOC_PA_Modelling_reversed: (1 item) for who participants are examples in sports
    SOC_PA_Modelling: (1 item) who are exercising in a way participants also want to exercise
    --------------------------------------------------------------
    formula should be a dictionary with the variables used and the weights for each of them. For instance:
    {
        SOC_GEN_Advice: 1,
        SOC_GEN_Friendship: 1,
        SOC_GEN_Leader: 1,
        SOC_GEN_INNOV: 0.5,
        SOC_DI_Com_network: 0.5,
        SOC_Di_Modelling_reversed: 0.2
    }
    '''
    # List with all the participants in the experiment
    pp = pd.read_csv('pp.csv', sep=';', header=0)
    list_participants = list(pp.Child_Bosse)

    # Read the file with the nominations from the participants and adapt the labels for the columns
    nominations = pd.read_csv('nominations.csv', sep=';', header=0)
    # nominations.columns = ['class', 'child', 'wave', 'variable', 'class_nominated', 'nominated', 'same_class']

    # Read formula to calculate the weight for the connections
    if formula_s is None:
        try:
            if label is None:
                formula = json.loads(open(('connections.json').format(level_f)).read())
            else:
                formula = json.loads(open(('connections_{}.json').format(level_f, label)).read())
        except Exception as ex:
            print(('File {}settings/connections_{}.json does not exist!').format(level_f, label))
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
    # Create the edges in the graph
    for node in connections_dict.items():
        destine = node[0]
        origins = node[1]
        for peer, weight in origins.items():
            if weight > 0:
                graph.add_edge(peer, destine, weight=weight)

    # Save the connections file in the results folder
    
    if label is None:
        connections_df.to_csv('connections.csv')
    else:
        connections_df.to_csv(('connections_{1}.csv').format(level_f, label))

    return graph

if __name__ == "__main__":
    # execute only if run as a script
    graph = generate_network()
    

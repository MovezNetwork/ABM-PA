
import numpy as np
import random
import networkx as nx
import pandas as pd
from xlwt import Workbook
from xlrd import open_workbook
from xlutils.copy import copy
import os
from src.models import diffuse_behavior_PA, contagion_model

def get_subgraphs_centrality(graph,centrality_type='indegree'):
    
    input_simulation = json.loads(open('../input/simulation.json').read())
    class_list = input_simulation['classes']

    # Create a dictionary. Keys are the classes, and the values are list of students
    class_dictionary = {}

    for c in class_list:
        class_dictionary[c] = []
    
    for node, key in graph.nodes.data('class'):
        class_dictionary[int(key)].append(node)

    list_subgraphs = [] 
    cent_dict = {}
    
    
    
    for c in class_list:
        #print(class_dictionary[c])
        subgraph = graph.subgraph(class_dictionary[c])
        list_subgraphs.append(subgraph)
        if centrality_type=='closeness':
            for key, cent in nx.closeness_centrality(subgraph).items():
                cent_dict[key]=cent
        elif centrality_type=='outdegree':
            for key, cent in nx.out_degree_centrality(subgraph).items():
                cent_dict[key]=cent
        elif centrality_type=='indegree':
            for key, cent in nx.in_degree_centrality(subgraph).items():
                cent_dict[key]=cent
        elif centrality_type=='betweenness':
            for key, cent in nx.betweenness_centrality(subgraph).items():
                cent_dict[key]=cent
        #centrality_dict = nx.degree_centrality(subgraph)
    # cent_dict is the dictionary with all centralities for all nodes
    return cent_dict, list_subgraphs


def get_class_dictionary(graph, level_f='../',centrality_type='indegree'):
    '''
    Generates the dictionary with BMI and env for each kid per class
    Create a dictionary. Keys are the classes, and the values are list 
        of students
    '''
    
    input_simulation = json.loads(open('../input/simulation.json').read())
    class_list = input_simulation['classes']
    
    
    class_dictionary = {}

    for c in class_list:
        class_dictionary[c] = []

    centrality_dict, _ = get_subgraphs_centrality(graph,level_f,centrality_type,class_list)

    for node, key in graph.nodes.data('class'):
        class_dictionary[int(key)].append((node,
                                           graph.nodes[node]['gender'],
                                           graph.nodes[node]['bmi_cat'],
                                           graph.nodes[node]['env'],
                                           centrality_dict[node],
                                           graph.nodes[node]['bmi'],
                                           graph.nodes[node]['PA']))
    return class_dictionary


def apply_intervention(graph, perc = 0, intervention = '', debug=False):
    
    
    if(intervention == 'outdegree' or intervention == 'indegree' or intervention == 'closeness' or intervention == 'betweenness'):
        selected_nodes = apply_interventions_centrality(graph,perc,centrality_type = intervention)
    elif(intervention == 'max'  or intervention == 'min'):
        selected_nodes = apply_intervention_pal(graph,perc,criteria = intervention)
    elif(intervention == 'random'):
        selected_nodes = apply_intervention_random_nodes(graph,perc)
    elif(intervention == 'highrisk'):
        selected_nodes = apply_interventions_high_risk(graph,perc)
    elif(intervention == 'vulnerability'):
        selected_nodes = apply_interventions_vulnerability(graph,perc)
    elif(intervention == 'nointervention'):
        return graph
    
    '''
    Apply the intervention for the PA
    '''
    for node in selected_nodes:
        # 17%
        if debug:
            print('Node #{} - old PA: {}'.format(node,graph.nodes[node]['PA']))
        graph.nodes[node]['PA'] = graph.nodes[node]['PA']*1.17
        graph.nodes()[node]['PA_hist'] = [graph.nodes()[node]['PA']]
        if debug:
            print('Node #{} - new PA: {}'.format(node,graph.nodes[node]['PA']))
            
    return graph


def apply_intervention_random_nodes(graph, perc=0.1, level_f='../', debug=False):
    '''
    Random selection of nodes based purely in the percentage
    '''
    
    list_selected = []
    class_dictionary = get_class_dictionary(graph, level_f)

#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        #print(c, round(len(data)*perc))
        num_selected = round(len(data)*perc)

        total = len(data)
        children = [item[0] for item in data]
        
        list_selected = list_selected + random.sample(children, num_selected)
        if debug:
            print('Class {}: #{} nodes'.format(c,num_selected))
            print('{0}'.format(list_selected))

    return list_selected
    
def apply_intervention_pal(graph, perc=0.1, level_f='../', criteria='min',debug=False):
    '''
    Random selection of nodes based purely in the percentage
    '''
    
    list_selected = []
    class_list=[graph.graph['class']]
    class_dictionary = get_class_dictionary(graph, level_f,class_list)

#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        #print(c, round(len(data)*perc))
        num_selected = round(len(data)*perc)

        total = len(data)

        # centrality_list : a list of tuples...
        centrality_list = [(item[0],item[6]) for item in data]
        if criteria=='max':
            centrality_list.sort(key=lambda tup: tup[1],reverse=True)
        elif criteria=='min':
            centrality_list.sort(key=lambda tup: tup[1])
            
            
        # all nodes just for information purposes....
        all_nodes=centrality_list
        all_nodes_id = [item[0] for item in all_nodes]      
        all_nodes_pal=[item[1] for item in all_nodes]

        
        
        
        
        selected_nodes = centrality_list[0:num_selected]
        selected_nodes_id = [item[0] for item in selected_nodes]      
        selected_nodes_pal=[item[1] for item in selected_nodes]
        
        list_selected = list_selected + selected_nodes_id
        

#         print('Class {}: #{} nodes'.format(c,num_selected))
#         print('{0}'.format(list_selected))
#         print('Selected nodes : {0}'.format(selected_nodes_id))
#         print('PAL selected nodes {0}'.format(selected_nodes_pal))
#         print('All nodes {0}'.format(all_nodes_id))
#         print('PAL all nodes {0}'.format(all_nodes_pal))

    return list_selected

def apply_interventions_centrality(graph, perc=0.1, level_f='../', debug=False, centrality_type='indegree'):
    '''
    Select nodes with higher centrality
    '''
    list_selected = []
    class_list=[graph.graph['class']]

    class_dictionary = get_class_dictionary(graph, level_f,class_list,centrality_type)
#     print(class_dictionary)
#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes for centrality intervention'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        
        num_selected = round(len(data)*perc)
        total = len(data)
        # Select the info about centrality and order the list
        centrality_list = [(item[0],item[4],item[1],item[6]) for item in data]
#         centrality_list = [(item[0],item[4]) for item in data]
        
        # centrality_list : a list of tuples...
        centrality_list.sort(key=lambda tup: tup[1],reverse=True)
    
        # all nodes just for information purposes....
        all_nodes=centrality_list
        all_nodes_id = [item[0] for item in all_nodes]      
        all_nodes_centrality=[item[1] for item in all_nodes]
        all_nodes_pal=[item[3] for item in all_nodes]
        all_nodes_gender=[item[2] for item in all_nodes]    
    
        
        selected_nodes_centrality,selected_nodes_id,selected_nodes_pal, selected_nodes_gender = getRandomNodes(num_selected,all_nodes_centrality,all_nodes_id,all_nodes_pal,all_nodes_gender)
        
            
        list_selected = list_selected + selected_nodes_id    

        

    return list_selected



def apply_interventions_high_risk(graph, perc=0.1, level_f='../', debug=False):
    '''
    Select nodes with higher BMI
    '''
    list_selected = []

    class_dictionary = get_class_dictionary(graph, level_f)
    #print(class_dictionary)
#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes for high risk intervention (BMI)'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        
        num_selected = round(len(data)*perc)
        total = len(data)
        # Select the info about centrality and order the list
        bmi_list = [(item[0],item[5]) for item in data]
        #print(bmi_list)
        bmi_list.sort(key=lambda tup: tup[1],reverse=True)
        if debug:
            print(bmi_list)
        selected_nodes = bmi_list[0:num_selected]
        selected_nodes = [item[0] for item in selected_nodes]
        list_selected = list_selected + selected_nodes    
        
        if debug:
            print('Class {}: #{} nodes'.format(c,num_selected))
            print('{0}'.format(selected_nodes))

    return list_selected


def apply_interventions_vulnerability(graph, perc=0.1, level_f='../', debug=False):
    '''
    Select nodes with higher BMI
    '''
    list_selected = []

    class_dictionary = get_class_dictionary(graph, level_f)
    #if debug:
    #    print(class_dictionary)
    #print(class_dictionary)
#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes for vulnerability intervention'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        
        num_selected = round(len(data)*perc)
        total = len(data)
        # Select the info about centrality and order the list
        env_list = [(item[0],item[3]) for item in data]
        #print(env_list)
        # The lower the worse the environment
        env_list.sort(key=lambda tup: tup[1],reverse=False)
        if debug:
            print(env_list)
        selected_nodes = env_list[0:num_selected]
        selected_nodes = [item[0] for item in selected_nodes]
        list_selected = list_selected + selected_nodes    
        
        if debug:
            print('Class {}: #{} nodes'.format(c,num_selected))
            print('{0}'.format(selected_nodes))

    return list_selected



# Max influence
def apply_intervention_max_influence(graph, perc=0.1, years=1, thres_PA = 0.2, I_PA = 0.00075, level_f='../', debug=False, modeltype='diffusion', delta=0.2):
    '''
    Objective is to maximize the PA of the whole network.
    '''
    
    all_selected = []
    class_dictionary = get_class_dictionary(graph, level_f)
    
#     print('------------------------------------------------------------------')
#     print('Getting {0}% of the nodes for maximum influence'.format(perc))
#     print('------------------------------------------------------------------')

    for c, data in class_dictionary.items():
        
#         print('\nClass {}: Starting'.format(c))
#         print('--------------------------------')
        num_selected = round(len(data)*perc)
        total = len(data)
        
        selected_nodes = []
        
        while len(selected_nodes) < num_selected:
            node_n=len(selected_nodes)+1
            if debug:
                print('Class {}: Selecting #{} node'.format(c, node_n))
            
            # All nodes in this subgraph
            list_nodes = [item[0] for item in data]
            
            # check the available nodes in the 
            available_nodes = list(set(list_nodes) - set(selected_nodes))
            
            impact_nodes = {}

            for node in available_nodes:
                # copy graph to reset the nodes
                g = graph.subgraph(list_nodes).copy()

                # append without altering selected nodes list...
                temp_list = selected_nodes + [node]
                apply_intervention(g, selected_nodes=temp_list, debug=False)
                if(modeltype=='diffusion'):
                    diffuse_behavior_PA(g, years=years, thres_PA=thres_PA, I_PA=I_PA)
                elif(modeltype=='contagion'):
                    contagion_model(g, years=years,delta=delta , model='original')
                # Calculate impact
                sum_diff_PA = 0

                for n in g.nodes():
                    #final_PA = nx.get_node_attributes(g,'PA_hist')[968]
                    final_PA = g.nodes()[n]['PA_hist'][-1]
                    initial_PA = g.nodes()[n]['PA_hist'][0]
                    
                    sum_diff_PA = sum_diff_PA + (final_PA - initial_PA)
                
                impact_nodes[node] = sum_diff_PA
            
            #print('Impact of all nodes:')
            #print(pd.Series(impact_nodes).sort_values(ascending=False))

            # Get the nodes sorted by the BW
            keys_sorted = sorted(impact_nodes, key=impact_nodes.get, reverse=True)
            
            selected_nodes.append(keys_sorted[0])
            all_selected.append(keys_sorted[0])
            if debug:
                print('Selected node: {}'.format(keys_sorted[0]))
            
    return apply_intervention(graph, selected_nodes=all_selected)



def getRandomNodes(numNodes,valueArray,idArray,palArray,genderArray):
    
    finalindices=[]
    sel=numNodes
    cent=valueArray
    ids=idArray
    pals=palArray
    genders=genderArray
    
    selhelp=sel
    centhelp=cent
    idshelp=ids
    palhelp=pals
    genderhelp=genders

    selectedcent=[]
    selectedids=[]
    selectedpal=[]
    selectedgender=[]
    
    while len(finalindices)!=sel: 

        test=get_max_indices(centhelp)

#         print('current cent: ' + repr(centhelp))
#         print('current ids: ' + repr(idshelp))
#         print('max indices: ' + repr(test))
#         print('selected '+repr([centhelp[i] for i in test]))



        if len(test)>selhelp:
            test=random.sample(test,selhelp)
            finalindices.extend(random.sample(test,selhelp))
#             print('1 finalindices: ' + repr(finalindices))
        elif len(test)==selhelp:
            finalindices.extend(test)

#             print('2 finalindices: ' + repr(finalindices))
        elif len(test)<selhelp:
            finalindices.extend(test)
#             print('3 finalindices: ' + repr(finalindices))

        selectedcent.extend([centhelp[i] for i in test])
        selectedids.extend([idshelp[i] for i in test])    
        selectedpal.extend([palhelp[i] for i in test])
        selectedgender.extend([genderhelp[i] for i in test]) 
        
        # remove from the list
        centhelp = [x for i,x in enumerate(centhelp) if i not in test]
        idshelp = [x for i,x in enumerate(idshelp) if i not in test]
        palhelp = [x for i,x in enumerate(palhelp) if i not in test]
        genderhelp = [x for i,x in enumerate(genderhelp) if i not in test]
#         print('finalindices: ' + repr(finalindices))
        selhelp=selhelp-len(test)
#         print('need to fill '+repr(selhelp))
#         print('*****************************')

#     print('FINAL RESULT')
  
    return selectedcent,selectedids, selectedpal, selectedgender


def get_max_indices(vals):
    maxval = None
    index = 0
    indices = []
    while True:
        try:
            val = vals[index]
        except IndexError:
            return indices
        else:
            if maxval is None or val > maxval:
                indices = [index]
                maxval = val
            elif val == maxval:
                indices.append(index)
            index = index + 1
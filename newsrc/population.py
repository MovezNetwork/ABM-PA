'''
Methods related to Population creation.
'''
import json
import networkx as nx
import numpy as np
import os
import pandas as pd
import random


class Population:
        def __init__(self, name):
            self.name = name

            
        def create_population(self):
            pass


        def assign_PA(self,metric='steps'):

            '''
            Assign physical activity value for nodes.

            Args:
                metric (str): physical activity metrics to use. default is number of steps.

            Returns:
                dictionary: Dictionary with average steps per child and per wave.
            '''            
            
            df_pal = pd.read_csv(self.input_args['agent_pal_file'], sep=';', header=0, encoding='latin-1')
            df_pal = df_pal[df_pal['Child'].isin(self.input_args['participants'])]

            df_pal = df_pal.groupby(['Child', 'Wave']).mean()['Steps'].reset_index()
            df_pal.Steps = df_pal.Steps * 0.000153
            df_pal = df_pal.pivot(index='Child', columns='Wave')['Steps']
            # Avg wave 5 data is assigned as PAL
            return dict(df_pal[5])

        def assign_PA_random(self, dict):
            '''
            Sample 408 physical activity levels based on data from wave 1
            '''

            # Read data
            fitbit_new = pd.read_csv(self.input_args['agent_pal_file'], sep=';', header=0)
            steps_mean_wave_new = fitbit_new.groupby(['Child', 'Wave']).mean()['Steps'].reset_index()
            steps_mean_wave_new.Steps = steps_mean_wave_new.Steps * 0.000153
            steps_mean_wave_new = steps_mean_wave_new.pivot(index='Child', columns='Wave')['Steps']

            steps_wave1 = steps_mean_wave_new[1]
            PAL_wave1 = steps_wave1.dropna()

            # number of agents
            n = len(dict.keys())

            # sample from data
            np.random.seed(1234)
            r = np.random.random_sample(n)
            pal = np.quantile(PAL_wave1, q=r)

            # assign to dictionary
            i = 0
            for key in dict.keys():
                dict[key] = pal[i]
                #print(i, "key", key, " r ", r[i], " pal ", pal[i] )
                i = i + 1

            return dict


        def assign_environment(self):


            '''
            Assign environment value for nodes. Combination of different questionnaire responses for owning computers, car, ownroom or allowing summer vacation.

            Returns:
                dictionary: Dictionary with environment score per child.
            '''

            env = pd.read_csv(self.input_args['agent_environment_file'], sep=';', header=0, encoding='latin-1')
            env = env[["Child","GEN_FAS_computer_A01" ,"GEN_FAS_car_A01", "GEN_FAS_vacation_A01", "GEN_FAS_ownroom_A01"]]

            env = env[env['Child'].isin(self.input_args['participants'])]


            env['FAS_Score_R'] = env['GEN_FAS_computer_A01'] + env['GEN_FAS_vacation_A01'] + env['GEN_FAS_car_A01']*1.5 + env['GEN_FAS_ownroom_A01']*3

            # To keep the values between 0 and 2.
            env.FAS_Score_R = abs((env.FAS_Score_R/6) - 2)
            env.index = env['Child']

            env_dict = dict(env['FAS_Score_R'])
            for key, value in env_dict.items():
                # to all the NaN assign the average FAS_Score
                if np.isnan(value):
                    env_dict[key] = np.float64(env.FAS_Score_R.mean())
        

            return env_dict


        def assign_bmi(self):

            '''
            Assign BMI value for nodes.

            Returns:
                dictionary: Dictionary with BMI score per child.
            '''
            bmi = pd.read_csv(self.input_args['agent_bmi_file'], sep=';', header=0, encoding='latin-1')
            bmi = bmi[bmi.Wave == 5]
            bmi = bmi[bmi.Child.isin(self.input_args['participants'])]
            bmi = bmi[['Child', 'BMI']]
            bmi.index = bmi.Child
            bmi = bmi['BMI']
            
            return dict(bmi)

        def fix_float64(self,orig_dict):

            '''
            Helper method converts the numpy.float64 values from a dictionary to native float type.

            Args:
                orig_dict (dict): original dictionary with fault floats

            Returns:
                dictionary: Dictionary with updated float values.
            '''

            new_dict = {}
            for k, item in orig_dict.items():
                try:    
                    new_dict[k] = -1.0 if np.isnan(item) else item.item()
                except:
                    new_dict[k] = -1.0


            return new_dict

        
        def assign_basic(self):

            '''
            Assign gender, age and class information per node.

            Returns:
                dictionary: Dictionary with gender, age and class information per child.
            '''


            pp = pd.read_csv(self.input_args['agent_personal_file'], sep=';', header=0, encoding='latin-1')
            pp = pp[pp.Child.isin(self.input_args['participants'])]
            gender_df = pp.groupby(['Child']).mean()['Sex']
            age_df = pp.groupby(['Child']).mean()['Age_W5']

            # Assign Class
            pp['class'] = pp.Class_Y3
            pp.index = pp.Child
            class_df = pp['class']

            return dict(gender_df), dict(age_df), dict(class_df)


        def get_empirical(self,metric='steps',classes=[]):
            '''
            Get empirical physical activity data. 

            Args:
                metric (str): physical activity metrics to use. default is number of steps.
                classes (array): list of class ids

            Returns:
                dataframe: physical activity data (steps) per child and wave.
            '''

            df_pal = pd.read_csv(self.input_args['agent_pal_file'], sep=';', header=0, encoding='latin-1')
            df_pal = df_pal[df_pal['Child'].isin(self.input_args['participants'])]

            df_pal = df_pal.groupby(['Child', 'Wave']).mean()['Steps'].reset_index()
            df_pal.Steps = df_pal.Steps * 0.000153
            df_pal = df_pal.pivot(index='Child', columns='Wave')['Steps']

            return df_pal
        


        def get_subgraphs_centrality(self,graph,centrality_type='indegree'):
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

            class_list = self.input_args['classes']

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


        def get_class_dictionary(self,graph,centrality_type='indegree'):

            '''
            Generates the dictionary with BMI and env for each kid per class
            Create a dictionary. Keys are the classes, and the values are list 
            of students


            TODO: This method is called multiple times. Suggestion: better to store output in an object, and reuse it. (perhaps in simulation)
            '''

            class_list = self.input_args['classes']
            class_dictionary = {}

            for c in class_list:
                class_dictionary[c] = []

            centrality_dict, _ = self.get_subgraphs_centrality(graph,centrality_type)

            for node, key in graph.nodes.data('class'):
                class_dictionary[int(key)].append((node,
                                                   graph.nodes[node]['gender'],
                                                   graph.nodes[node]['bmi_cat'],
                                                   graph.nodes[node]['env'],
                                                   centrality_dict[node],
                                                   graph.nodes[node]['bmi'],
                                                   graph.nodes[node]['PA']))
            return class_dictionary


        def select_influential_agents(self,graph, perc = 0, intervention = '', debug=False):
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

            if(intervention == 'outdegree' or intervention == 'indegree' or intervention == 'closeness' or intervention == 'betweenness'):
                selected_nodes = self.select_influential_agents_centrality(graph,perc,centrality_type = intervention)
            elif(intervention == 'max'  or intervention == 'min'):
                selected_nodes = self.select_influential_agents_pal(graph,perc,criteria = intervention)
            elif(intervention == 'random'):
                selected_nodes = self.select_influential_agents_random(graph,perc)
            elif(intervention == 'highrisk'):
                selected_nodes = self.select_influential_agents_high_risk(graph,perc)
            elif(intervention == 'vulnerability'):
                selected_nodes = self.select_influential_agents_vulnerability(graph,perc)
            elif(intervention == 'nointervention'):
                return (graph,[])

            '''
            Increase the PA by 1.17 for the selected influential agents (based on the gabrianeli paper)
            '''
            for node in selected_nodes:
                # 17%
                if debug:
                    print('Node #{} - old PA: {}'.format(node,graph.nodes[node]['PA']))
                graph.nodes[node]['PA'] = graph.nodes[node]['PA']*1.17
                graph.nodes()[node]['PA_hist'] = [graph.nodes()[node]['PA']]
                if debug:
                    print('Node #{} - new PA: {}'.format(node,graph.nodes[node]['PA']))

            return (graph,selected_nodes)


        def select_influential_agents_random(self, graph, perc=0.1, debug=False):
            '''
            Random Selection
            '''

            list_selected = []
            class_dictionary = self.get_class_dictionary(graph)
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
    
        def select_influential_agents_pal(self,graph, perc=0.1, criteria='min',debug=False):
            '''
            Selection based on PAL
            '''

            list_selected = []
            class_list=[graph.graph['class']]
            class_dictionary = self.get_class_dictionary(graph)
        
#             print('------------------------------------------------------------------')
#             print('Getting {0}% of the nodes'.format(perc))
#             print('------------------------------------------------------------------')

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


        def select_influential_agents_centrality(self,graph, perc=0.1, debug=False, centrality_type='indegree'):

            '''
            Select nodes with higher centrality
            '''

            list_selected = []
    
            class_list=[graph.graph['class']]
            class_dictionary = self.get_class_dictionary(graph,centrality_type)
        #     print('------------------------------------------------------------------')
        #     print('Getting {0}% of the nodes for centrality intervention'.format(perc))
        #     print('------------------------------------------------------------------')
            for c, data in class_dictionary.items():
                num_selected = round(len(data)*perc)
                total = len(data)
                # Select the info about centrality and order the list
                centrality_list = [(item[0],item[4],item[1],item[6]) for item in data]
        
                #first sort by centrality measure, then by pal
                centrality_list = sorted(centrality_list, key = lambda x: (-x[1], -x[3]))
#                 print('NEW sorted', centrality_list)
                selected_nodes = centrality_list[0:num_selected]
                selected_nodes_id = [item[0] for item in selected_nodes]
#                 print('NEW selected_nodes', selected_nodes)                
                list_selected = list_selected + selected_nodes_id  
                
            return list_selected


        def select_influential_agents_high_risk(self,graph, level_f='../', debug=False):

            '''
            Select nodes with higher BMI
            '''

            list_selected = []

            class_dictionary = self.get_class_dictionary(graph)
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


        def select_influential_agents_vulnerability(self,graph, perc=0.1, debug=False):

            '''
            Select nodes with higher BMI
            '''

            list_selected = []

            class_dictionary = self.get_class_dictionary(graph)
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
        def select_influential_agents_max_influence(self,graph, perc=0.1, years=1, thres_PA = 0.2, I_PA = 0.00075, debug=False, modeltype='diffusion', delta=0.2):

            '''
            Objective is to maximize the PA of the whole network.
            '''

            all_selected = []
            class_dictionary = self.get_class_dictionary(graph)

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
                        self.get_intervention_nodes(g, selected_nodes=temp_list, debug=False)
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

            return get_intervention_nodes(graph, selected_nodes=all_selected)


        def getRandomNodes(self,numNodes,valueArray,idArray,palArray,genderArray):
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

                test=self.get_max_indices(centhelp)

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


        def get_max_indices(self,vals):
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


        def get_class_graphs(self,graph):

            '''
            Return list of NetworkX subgraphs (each subgraph is a separate class).

            Specify wanted classes in c_list, generate all classes by default.

            writeToFile - set to True, if you want to generate gephi compatible files. Files are saved on the local path under
                          Subgraphs directory.

            label - the network type of the input graph.
            '''


            class_list = self.input_args['classes']
            writeToFile = self.input_args['writeToExcel'] 
            label = self.input_args['network']
            label = label[0]

        #     if(writeToFile):
        #         directory='output/ClassesSummary/GephiSubgraphs'
        #         if not os.path.exists(directory):
        #             os.makedirs(directory)
            # if list is empty, we want all the classes

            class_dictionary = {}

            for c in class_list:
                class_dictionary[c] = []

            for node, key in graph.nodes.data('class'):
                if key in class_dictionary: 
                    class_dictionary[int(key)].append(node)

            list_subgraphs = [] 
            cent_dict = {}

            for c in class_list:
                subgraph = graph.subgraph(class_dictionary[c]).copy()
                subgraph.graph['networkType']=label
                subgraph.graph['class']=c

                if(writeToFile):
                        directory='../output/Class'+repr(int(c))
                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        g_file = directory+'/class_' + repr(c) + '_' + label + '.gexf'
                        try:
                            nx.write_gexf(subgraph, g_file)
                        except IOError as e:
                            errno, strerror = e.args
                            print("I/O error({0}): {1}".format(errno,strerror))

                list_subgraphs.append(subgraph)

            return  list_subgraphs
        
        
        def population_network_summary(self,graph):
                        
            population_list = []
            node_data_list = []
            for subgraph in self.get_class_graphs(graph):
            
                dict_out_degree = dict(subgraph.out_degree())
                dict_in_degree = dict(subgraph.in_degree())
                dict_eigen_vector = dict(nx.eigenvector_centrality(subgraph))
                dict_closeness = dict(nx.closeness_centrality(subgraph))
                dict_betweenness = dict(nx.betweenness_centrality(subgraph))
                
                total_agents = subgraph.number_of_nodes()

                # graph centralization measures - on class level
                max_ind=max(dict_in_degree.values())
                dividor_ind=(len(dict_in_degree.values())-1)*(max_ind-1)
                sum_ind=0 

                max_outd=max(dict_out_degree.values())
                dividor_outd=(len(dict_out_degree.values())-1)*(max_outd-1)
                sum_outd=0   

                max_close=max(dict_closeness.values())
                dividor_close=(len(dict_closeness.values())-1)*(max_close-1) if (len(dict_closeness.values())-1)*(max_close-1)!=0 else 1
                sum_close=0

                max_beetwn=max(dict_betweenness.values())
                dividor_beetwn=(len(dict_betweenness.values())-1)*(max_beetwn-1)
                sum_beetwn=0            

                gender_f = 0
                gender_m = 0
                
                avg_envorinment_score = 0
                avg_bmi_score = 0
                
                isolated_nodes = []
                
                for nodedata in subgraph.nodes().data():
                    #for graph-level data
                    
                    #centralization measures
                    sum_ind = sum_ind+(max_ind-dict_in_degree[nodedata[0]])
                    sum_outd = sum_outd+(max_outd-dict_out_degree[nodedata[0]])
                    sum_close = sum_close+(max_close-dict_closeness[nodedata[0]])
                    sum_beetwn = sum_beetwn+(max_beetwn+dict_betweenness[nodedata[0]])
                    
                    #other measures
                    if(nodedata[1]['gender']==1.0):
                        gender_f = gender_f + 1
                        
                    #calculating the average weight    
                    num_edges = len(subgraph.edges(nodedata[0],data=True))
                    avg_weight = 0
                    if(num_edges > 0):
                        for (u, v, wt) in subgraph.edges(nodedata[0],data=True):
                            avg_weight = avg_weight +  wt['weight']

                        avg_weight = avg_weight/num_edges    
                        
                    avg_envorinment_score = avg_envorinment_score + nodedata[1]['env']    
                    avg_bmi_score = avg_bmi_score + nodedata[1]['bmi'] 
                    #isolated nodes - based on in-degree for now.
                    if dict_in_degree[nodedata[0]]==0:
                        isolated_nodes.append(nodedata[0])
                        
                    #participant-level data
                    node_data_list.append([nodedata[0], nodedata[1]['class'], nodedata[1]['gender'], nodedata[1]['PA'], nodedata[1]['bmi'],nodedata[1]['env'],dict_in_degree[nodedata[0]], dict_out_degree[nodedata[0]],dict_eigen_vector[nodedata[0]],dict_closeness[nodedata[0]],dict_betweenness[nodedata[0]],avg_weight])
                
                #population level data
                population_list.append([nodedata[1]['class'],total_agents,int((gender_f/total_agents)*100),subgraph.number_of_edges(), round(nx.density(subgraph),2), len(isolated_nodes),round(sum_ind/dividor_ind,2),round(sum_outd/dividor_outd,2),round(sum_close/dividor_close,2),round(sum_beetwn/dividor_beetwn,2),round(nx.degree_assortativity_coefficient(subgraph),2),round(avg_envorinment_score/total_agents,2),round(avg_bmi_score/total_agents,2)])
                
            
            #create the dataframes
            df_population_details = pd.DataFrame(population_list, columns = ["SchoolClassID", "NumberOfAgents", "PercentageFemale", "NumberConnections", "Density", "IsolatedNodes","CentralizationInDegree", "CentralizationOutDegree", "CentralizationCloseness", "CentralizationBetweenness", "DegreeAssortativity","AverageEnvironmentScore","AverageBMIScore"])
            
            df_agent_details = pd.DataFrame(node_data_list, columns = ["ParticipantID","SchoolClassID", "Gender", "PA", "BMI", "Environment", "InDegree", "OutDegree", "EigenVector", "Closeness", "Betweenness","Average_Weight"])
            
            if self.input_args['writeToExcel']:
                df_population_details.to_excel('../output/population_details.xlsx')
                df_agent_details.to_excel('../output/agents_details.xlsx')
        
            df_population_details = df_population_details.sort_values(by=['SchoolClassID'])        
            df_agent_details = df_agent_details.sort_values(by=['ParticipantID'])
            
            return df_population_details, df_agent_details

        def get_bmi_cat(self,gender,age,bmi):
    
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
        
        
class PeerNominatedDataPopulation(Population):
    
    def __init__(self,name,input_args):
        self.name = name
        self.input_args = input_args
        self.graph = self.create_population(nx.DiGraph())

    def create_population(self,graph):

                '''
                Create graph's weights between edges depending on the chosen formula. 
                The connections are created and weighted based on questionnaire responses. 
                Use all (the whole set of questions), gen (only the general questions) or friends (only the friends questions) to create the weights

                Args:
                    graph (Graph): input graph
                    label (str): label of the graph- gen, all or friends graph

                Returns:
                Graph: updated graph with weighted connections.

                '''

                # CREATE CONNECTIONS
                graph = graph
                formula = ''

                list_participants = self.input_args['participants']
                label = self.input_args['network']
                class_list = self.input_args['classes']


                nominations = pd.read_csv(self.input_args['peer_nomination_file'], sep=';', header=0, encoding='latin-1')
                nominations = nominations[nominations.Child.isin(list_participants)]
                nominations = nominations[nominations.Alter.isin(list_participants)] 

                # Read formula to calculate the weight for the connections
                try:
                    if label is None:
                        formula = json.loads(open('../input/connections.json').read())
                    elif label=='all':
                        formula = json.loads(open('../input/connections_all.json').read())
                    elif label=='gen':
                        formula = json.loads(open('../input/connections_gen.json').read())
                    elif label=='friend':
                        formula = json.loads(open('../input/connections_friend.json').read())
                except Exception as ex:
                    print('File {}settings/connections_{}.json does not exist!')
                    print(ex)
                    return

                # Sum of all weights from the formula
                max_score = sum(formula.values())

                # Create a dictionary with the connections and weights
                connections_dict = {}
                for child in list(list_participants):
                    connections_dict[child] = {}

                # To avoid repetition of nominations in different waves
                nominations_list = []

                for line in nominations[['Child', 'Alter', 'Variable']].iterrows():
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


                # POPULATE THE AGENTS
                PA_dict = self.assign_PA(metric='steps')
                if(self.input_args['assign_PA_random']):
                    PA_dict = self.assign_PA_random(PA_dict)

                gender_dict, age_dict, class_dict = self.assign_basic()
                environment_dict = self.assign_environment()
                bmi_dict = self.assign_bmi()

                PA_dict = self.fix_float64(PA_dict)
                gender_dict = self.fix_float64(gender_dict)
                age_dict = self.fix_float64(age_dict)
                class_dict = self.fix_float64(class_dict)
                environment_dict = self.fix_float64(environment_dict)
                bmi_dict = self.fix_float64(bmi_dict)

                nx.set_node_attributes(graph, values=PA_dict, name='PA')
                nx.set_node_attributes(graph, values=gender_dict, name='gender')
                nx.set_node_attributes(graph, values=age_dict, name='age')
                nx.set_node_attributes(graph, values=class_dict, name='class')
                nx.set_node_attributes(graph, values=environment_dict, name='env')
                nx.set_node_attributes(graph, values=bmi_dict, name='bmi')

                # Adding category for the nodes
                obesity_class = {}
                for node in graph.nodes():
                    obesity_class[node] = self.get_bmi_cat(gender_dict[node], age_dict[node], bmi_dict[node])

                nx.set_node_attributes(graph, values=obesity_class, name='bmi_cat')



                # Save the connections file in the results folder

    #             if label is None:
    #                 connections_df.to_csv('../output/connections.csv')
    #             else:
    #                 connections_df.to_csv(('../output/connections_{1}.csv').format(label))

                return graph


class CommunicationDataPopulation(Population):
    
    def __init__(self,name,input_args):
        self.name = name
        self.input_args = input_args
        self.graph = self.create_population(nx.DiGraph())

    def create_population(self,graph):

            '''
            Create graph's weights between edges depending on the chosen formula. 
            The connections are created and weighted based on questionnaire responses. 
            Use all (the whole set of questions), gen (only the general questions) or friends (only the friends questions) to create the weights

            Args:
                graph (Graph): input graph
                label (str): label of the graph- gen, all or friends graph

            Returns:
            Graph: updated graph with weighted connections.

            '''
            graph = graph
            classes = self.input_args['classes']
            
            df_comm = pd.read_csv(self.input_args['communications1_file'], sep=';', header=0, encoding='latin-1')
            df_comm = df_comm.append(pd.read_csv(self.input_args['communications2_file'], sep=';', header=0, encoding='latin-1'))
            df_comm = df_comm[df_comm.Type=="my"]
            df_comm = df_comm[df_comm.Class.isin(classes)]
            df_comm = df_comm[df_comm.Child.isin(self.input_args['participants'])]


            df_comm = df_comm.astype({'Participant': 'float64'})

            df_participants = df_comm.groupby(['Class','Participant','Child'])['UMID'].count().reset_index()
            df_participants = df_participants[['Class','Participant','Child']]
            
            #create the connections - edges
            for cl in classes:
                # the weight is according to version 1 
                df_temp = df_comm[df_comm.Class.isin([cl])]
                df_temp = df_temp.groupby(["Participant","RecipientChild"])["UMID"].count().reset_index(name='NumberMessages')
                df_temp['Weight'] = df_temp['NumberMessages']/max(df_temp['NumberMessages'])
                children_in_class = df_temp.Participant.unique()

                for index, row in df_temp.iterrows():
                    #only if the recepient is member of the same class - excluded intraclass communication
                    # this is not 100% ok, I need a list of all participants per class!
                    if(row[1] in children_in_class):
                        graph.add_edge(df_participants.loc[df_participants['Participant'] == row[0], 'Child'].iloc[0],df_participants.loc[df_participants['Participant'] == row[1], 'Child'].iloc[0],weight=row[3])
    
            # POPULATE THE AGENTS
            PA_dict = self.assign_PA(metric='steps')
            if (self.input_args['assign_PA_random']):
                PA_dict = self.assign_PA_random(PA_dict)

            gender_dict, age_dict, class_dict = self.assign_basic()
            environment_dict = self.assign_environment()
            bmi_dict = self.assign_bmi()

            PA_dict = self.fix_float64(PA_dict)
            gender_dict = self.fix_float64(gender_dict)
            age_dict = self.fix_float64(age_dict)
            class_dict = self.fix_float64(class_dict)
            environment_dict = self.fix_float64(environment_dict)
            bmi_dict = self.fix_float64(bmi_dict)

            nx.set_node_attributes(graph, values=PA_dict, name='PA')
            nx.set_node_attributes(graph, values=gender_dict, name='gender')
            nx.set_node_attributes(graph, values=age_dict, name='age')
            nx.set_node_attributes(graph, values=class_dict, name='class')
            nx.set_node_attributes(graph, values=environment_dict, name='env')
            nx.set_node_attributes(graph, values=bmi_dict, name='bmi')
            # Adding category for the nodes
            obesity_class = {}
            for node in graph.nodes():                
                obesity_class[node] = self.get_bmi_cat(gender_dict[node], age_dict[node], bmi_dict[node])

            nx.set_node_attributes(graph, values=obesity_class, name='bmi_cat')

                
                
            return graph   


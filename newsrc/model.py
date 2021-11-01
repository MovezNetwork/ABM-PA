import pandas as pd

class Model:
    def __init__(self,name):
        self.name = name
    
    def execute(self):
        pass   
    
    def validation(self):
        pass   
    
class DiffusionModel(Model):
    
    def __init__(self, name, input_args):
        self.name = name
        self.input_args = input_args
        self.thres_PA = self.input_args['thres_PA']
        self.I_PA = self.input_args['I_PA']
      
    
    def execute(self, graph, t):
        if t == 0:
            # Initiate hist vectors
            for node in graph.nodes():
                graph.nodes()[node]['PA_hist'] = [graph.nodes()[node]['PA']]
            
        else:
            for node in graph.nodes():
                # Cummulative influence from neighbors for PA and EI
                inf_PA = 0

                # Current values 
                PA = graph.nodes()[node]['PA_hist'][t-1]
                env = graph.nodes()[node]['env']

    #             if t == 1:
    #                 print('Node ID:' + repr(node))

                # Predecessors are the in-edges for each node
                sum_weights = 0
                # alternatives: successors predecessors
                for pred in graph.successors(node):

    #                 if t == 1:
    #                     print(' successors: ' + repr(pred) +'  weight:'+ repr(graph.edges[pred,node]['weight']))

                    w_pred2node = graph.edges[node,pred]['weight']
                    sum_weights = sum_weights+w_pred2node
                    inf_PA = inf_PA + (graph.nodes()[pred]['PA_hist'][t-1] - PA)*w_pred2node

    #             if t == 1:
    #                 print('*************')
                # Combined influence
                try:
                    inf_PA = inf_PA/sum_weights
                except:
                    inf_PA = 0

                # 2
                #if env == 0:
                #    env = 0.0001
                if env <= 0.1:
                    env = 0.1

                inf_PA_env = inf_PA / env if inf_PA >= 0 else inf_PA * env

                # 3
                # thres_PA_h, thres_PA_l, I_PA

                # For PA
                '''
                |inf_PA_env| <= thres_PA_l  or thres_PA_h <= |inf_PA_env|
                '''
                if (abs(inf_PA_env) <= self.thres_PA):
                    # Do nothing
                    PA_new = PA
                else:
                    if inf_PA_env > 0:
                        PA_new = PA * (1 + self.I_PA)
                    elif inf_PA_env < 0:
                        PA_new = PA * (1 - self.I_PA)
                    else:
                        PA_new = PA

                graph.nodes()[node]['PA_hist'].append(PA_new)

        return graph   

    def setThresholdPA(self, thres_PA_new):
        self.thres_PA = thres_PA_new

    def setIPA(self, I_PA_new):
        self.I_PA = I_PA_new

    def get_empirical_pa_data():
        '''
        Get empirical physical activity data. 

        Returns:
        dataframe: physical activity data (steps) per child and wave.

        TODO: method not complete
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


    def validation(self,df_interventions={}):
        '''
        TODO: method not complete.
        '''
        classes = self.input_args['classes']
        df_interventions = df_interventions
        emp_all=get_empirical_pa_data()[4]
        diffusion_all=pd.Series()
        
        for c in clist:
            for r_dict in df_interventions:
                if(r_dict['class']==c):
                    results_dict=r_dict
                    results_dict=results_dict['diffusion']['nointervention']['gen'][15].T
                    results_dict = results_dict[364]
                    diffusion_all=diffusion_all.append(results_dict)
                    
        to_drop=emp_all.copy().drop(diffusion_all.index.values).index.values
        index_diff=diffusion_all.index.values
        emp_all=emp_all.drop(to_drop)
        
        return mean_absolute_error(emp_all, diffusion_all)
        
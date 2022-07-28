'''
'''
import pandas as pd

class Model:
    '''
    Base Model Class.
    '''    
    def __init__(self,name):
        self.name = name
    
    def execute(self):
        pass   
    
    def validation(self):
        pass   
    
class DiffusionModel(Model):
    '''
    Diffusion Model Class.
    '''     
    def __init__(self, input_args):
        self.input_args = input_args
        self.thres_PA = self.input_args['thres_PA']
        self.I_PA = self.input_args['I_PA']
      
    
    def execute(self, graph, t):
        '''
        Diffusion model execution method.  

        Args:
            graph (Graph): input graph
            t (Integer): timepoint of a simulation (in days)

        Returns:
            NetworkX DiGraph: graph with updated PAL values based on the diffusion model run.
        '''

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
        '''
        Setter method: thres_PA model parameter

        Args:
            thres_PA_new (Float): new PA_threshold model parameter
        
        '''
        self.thres_PA = thres_PA_new

    def setIPA(self, I_PA_new):
        '''
        Setter method: I_PA model parameter  

        Args:
            thres_PA_new (Float): new I_PA model parameter  
        
        '''
        self.I_PA = I_PA_new
        
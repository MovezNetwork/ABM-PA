from src.graph import generate_network_PA,get_empirical
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from random import *
from src.classes_analysis import get_class_graphs

import networkx as nx
import numpy as np
import pandas as pd
from pprint import pprint
from random import random
from time import time


def get_graphs_PA_df(graph):
    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']
    return pd.DataFrame(PA_dict).mean(axis=1)

def contagion_model(graph,years=1,delta = 0.1, model = 'original'):
    #static values
#     openn = 0.5
#     exp = 0.5

    # calculate the expressiveness and openness
    t=graph.graph
    classID=t['class']
    
    print(classID)
    print(delta)
    print('*********')
    eo=expr_open(classID)
    avge=eo.expr.mean()
    avgo=eo.open.mean()
    
    for t in range(round(365*years)):
        # Initialize the data on day 0
        if t == 0:
            # Initiate hist vectors
            for node in graph.nodes():
                graph.nodes()[node]['PA_hist'] = [graph.nodes()[node]['PA']] 
                
            continue

        #go through each agent in the social graph
        for node in graph.nodes():
            # Coefficient to calculate the aggimpact
            cs = 0.0
            aggimpact = 0.0
            speed_factor = 0.0
            num_neighbours=graph.in_degree(node)
#             openn=0.5
            openn=eo.at[node,'open'] if eo.at[node,'open']!=0 else avgo
            # Current values
            PA = graph.nodes()[node]['PA_hist'][t-1]

            if(num_neighbours>0):
                # Neighbors are the in-edges (specified  by the predecessors)
                for neighbour in graph.predecessors(node):
#                     exp=0.5
                    exp=eo.at[neighbour,'expr'] if eo.at[neighbour,'expr']!=0 else avge
                    connect = graph.edges[neighbour,node]['weight']
                    
                    # connection weight for each neighboor
                    wba = exp * connect * openn
                    # speed factor is a sum of all the neighbour connection weights
                    speed_factor = speed_factor + wba
                    # we need this for the aggregated impact below
                    cs = cs + exp * connect
                    
                # calculate the aggregate impact
                for neighbour in graph.predecessors(node):
#                     exp=0.5
                    exp=eo.at[neighbour,'expr'] if eo.at[neighbour,'expr']!=0 else avge
                    connect = graph.edges[neighbour,node]['weight']  # graph.get_edge_data(neigh, node).values()[0]
                    try:
                        # It has a copy of the previous date - taking the second argument of the tuple ([1])
                        pal_neigh = graph.nodes()[neighbour]['PA_hist'][t-1]
                        #print('Node' + repr(node)+'Edge' + repr(pred) + 'PAL '+repr(v_neigh))
                    except:
                        print('Node ' + repr(neighbour))
                    if cs == 0:
                        aggimpact = 0
                    else:
                        aggimpact = aggimpact + (exp * connect * pal_neigh) / cs

                try:
                    old_pal = graph.nodes()[node]['PA_hist'][t-1]
                except:
                    print("Node "+repr(node)+"OLD PALs "+ repr(graph.nodes()[node]['PA_hist'][t-1]))
                
                            # Definition of the speed factor
                if model == 'original':
                    new_pal = old_pal + speed_factor * (aggimpact - old_pal) * delta
                elif model == 'weighted':
                    if num_neighbours == 0:
                        new_pal = old_pal
                    else:
                        new_pal = old_pal + (speed_factor / num_neighbours) * (aggimpact - old_pal) * delta
                elif model == 'logistic':
                    new_pal = old_pal + logistic(speed_factor) * (aggimpact - old_pal) * delta
                else:
                    print("Wrong value for model!")
                
                new_pal = old_pal + speed_factor * (aggimpact - old_pal) * delta
                graph.nodes()[node]['PA_hist'].append(new_pal)
            else:
                graph.nodes()[node]['PA_hist'].append(PA)
                   
    return graph


def logistic(number):
    steepness = 0.3
    threshold = 20
    log_number = (1 / (1 + np.exp(-steepness * (number - threshold))) - 1 / (1 + np.exp(steepness * threshold))) * \
                (1 + np.exp(-steepness * threshold))
    return log_number

def fix_float64(item):
    '''
    This function converts the numpy.float64 values from a dictionary to native float type.
    {k: np.asscalar(item) for k, item in orig_dict.items()}
    '''
    try:    
        newitem = -1.0 if np.isnan(item) else np.asscalar(item)
    except:
        newitem = -1.0
            #print(k, item)
    return newitem

def get_graphs_PA_df(graph):
    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']
    return pd.DataFrame(PA_dict).mean(axis=1)

def get_graphs_PA_df_detailed(graph):
    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']
    return pd.DataFrame(PA_dict)

def diffuse_behavior_PA(graph, years=1, thres_PA = 0.2, I_PA = 0.00075):
    '''
    Should run the social contagion for the PAs in the model.
    |- Inputs
        |-- years: amount of time for the simulation
        |-- thres_PA: threshold for the PA to take effect
        |-- I_PA: speed factor for PA for the changes
    '''
    for t in range(round(365*years)):
        # Initialize the data on day 0
        if t == 0:
            # Initiate hist vectors
            for node in graph.nodes():
                graph.nodes()[node]['PA_hist'] = [graph.nodes()[node]['PA']]
            continue

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
            if (abs(inf_PA_env) <= thres_PA):
                # Do nothing
                PA_new = PA
            else:
                if inf_PA_env > 0:
                    PA_new = PA * (1 + I_PA)    
                elif inf_PA_env < 0:
                    PA_new = PA * (1 - I_PA)
                else:
                    PA_new = PA

            graph.nodes()[node]['PA_hist'].append(PA_new)

    return graph


def mapS_I(val):
    
    ret=0
    if val == '1':
        ret = 0
    elif val == '2':
        ret = 0.2
    elif val == '3':
        ret = 0.4
    elif val == '4':
        ret = 0.6
    elif val == '5':
        ret = 0.8
    elif val == '6':
        ret = 1
    
    return ret

def mapSelfDesc(listTraits,oldEx,oldOp):
    
    total=len(listTraits)
    
    #count the traits related to op, ex
    op=0
    ex=0
    rop=0
    
    #new values for Ex and Op
    addEx=0
    addOp=0
    
    childID=listTraits[0]
#     print(childID)
    # per person traits
    for desc in listTraits:
        if desc=='Avontuurllijk':
            op=op+1        
        if desc=='Sociaal':
            op=op+1 
        if desc=='Creatief':
            op=op+1    
        if desc=='Behulpzaam':
            ex=ex+1      
        if desc=='Stil':
            rop=rop+1
        if desc=='Populair':
            op=op+1
        if desc=='Verlegen':
            rop=rop+1
        if desc=='Enthousiast':
            ex=ex+1        
        if desc=='Zelfverzekerd':
            ex=ex+1        
        if desc=='Chill':
            ex=ex+1         
        if desc=='Agressief':
            ex=ex+1
        if desc=='Cool':
            ex=ex+1
        if desc=='Dapper':
            op=op+1        
        if desc=='Bazig':
            ex=ex+1
        if desc=='Aantrekkelijk':
            ex=ex+1
        if desc=='Grappig':
            ex=ex+1        
        if desc=='Rustig':
            rop=rop+1        
        if desc=='Rolmodel':
            ex=ex+1         
        if desc=='Leider':
            ex=ex+1            
        if desc=='Leergierig':
            op=op+1        
        if desc=='Nieuwsgierig':
            op=op+1         
        if desc=='Vernieuwend':
            op=op+1
            
    # how much traits we mapped        
    totalMapped=op+ex+rop
    
    
    # max addition/removal can be +-0.25, can also change to +-0.5 but maybe is too much?
    if(totalMapped>0 and totalMapped<=5):
        changeE=ex/20
        changeO=(op-rop)/20

        addEx=changeE if changeE<=0.25 else 0.25
        addOp=changeO if changeO<=0.25 else 0.25
           
    elif totalMapped>5:
        #ratio
        x=totalMapped/5
        #how much  of the 5 to assign to expressiveness and how much to openness
        e=ex
        o=op+rop
       
        ae=int(round(ex/x))
        ao=int(round(o/x))
        
        if(e==o):
            rand=randint(1, 2)
            if rand==1:
                ae=ae+1
            elif rand==2:
                ao=ao+1
                
            

#         print(repr(childID)+ ' e:' + repr(e)+ ' o:' + repr(o))
#         print(repr(ae) + ' + ' + repr(ao))
        
        addEx=ae/20 if ae/20<=0.25 else 0.25
        addOp=ao/20 if ao/20<=0.25 else 0.25
        
    #print(repr(childID)+' addEx '+repr(addEx)+' addOp '+repr(addOp))    
    return oldEx+addEx,oldOp+addOp


def allChildrenInClass():
    pp = pd.read_csv('../data/pp.csv', sep=';', header=0)
    pp['cl'] = pp.Class_Y1
    pp['child'] = pp.Child_Bosse
    # Fill the missing data at Class column with the data from Y1.
    pp['cl'].fillna(pp.Class_Y2, inplace=True)
    pp.index = pp.Child_Bosse
    class_df = pp[['cl','child']]
    class_list=class_df['cl'].tolist()
    
    wanted=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
    class_df=class_df[class_df.cl.isin(wanted)]
    allChildrenClass = pd.DataFrame(columns=['Class','NumChildren'])
    for u in wanted:
        occurance=class_list.count(u)
        allChildrenClass.loc[len(allChildrenClass)] = [u,occurance]
        allChildrenClass.to_csv('classAllChildren.csv')
     
    return class_df


def model_validation_mse(model='contagion',df_int={}):
    clist=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
    emp_all=get_empirical()[4]
    df_interventions=df_int
    if model=='contagion':
        contagion_all=pd.Series()
        for c in clist:
            for r_dict in df_interventions:
                if(r_dict['class']==c):
                    results_dict=r_dict
                    results_dict=results_dict['contagion']['nointervention']['gen'][15].T
                    results_dict = results_dict[364]
                    contagion_all=contagion_all.append(results_dict)
                    
        to_drop=emp_all.copy().drop(contagion_all.index.values).index.values
        index_cont=contagion_all.index.values
        emp_all=emp_all.drop(to_drop)
        return mean_absolute_error(emp_all, contagion_all)
        
    elif model=='diffusion':    
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
    
    else:
        return 100

def expr_open(classID):
    #get all traits dataframes
    traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
    ids=traits.filter(regex=("ID"))
    
    # Nutri Questions
    nutrid=traits.filter(regex=("DI_NutriLoC_W019_D03_I01_LoC.*"))
    nutrid=pd.concat([ids,nutrid],axis=1)
    
    nutri_c_list=[]
    nutri_c_list.append('ID')
    for x in range(1, len(nutrid.columns)):
        nutri_c_list.append('a'+str(x))
    nutrid.columns = nutri_c_list
    #print(repr(len(nutrid.columns)) + ' nutri')

    #Need Belong
    needbeld=traits.filter(regex=("Gen_Need2belong_W019_D02_I01_NtB.*"))
    needbeld=pd.concat([ids,needbeld],axis=1)
    #print(repr(len(needbeld.columns)) + ' needbeld') 
    
    needbel_c_list=[]
    needbel_c_list.append('ID')
    for x in range(1, len(needbeld.columns)):
        needbel_c_list.append('a'+str(x))
    needbeld.columns = needbel_c_list

    #Pro Social
    socd=traits.filter(regex=("GEN_prosocial_W019_D06_I01_GEN_prosocial.*"))
    socd=pd.concat([ids,socd],axis=1)
    #print(repr(len(socd.columns)) + ' socd') 
    
    soc_c_list=[]
    soc_c_list.append('ID')
    for x in range(1, len(socd.columns)):
        soc_c_list.append('a'+str(x))
    socd.columns = soc_c_list
    
    #Self-Esteem
    selfestd=traits.filter(regex=("GEN_Selfesteem_W019_D04_I01_S.*"))
    selfestd=pd.concat([ids,selfestd],axis=1)
    
    selfest_c_list=[]
    selfest_c_list.append('ID')
    for x in range(1, len(selfestd.columns)):
        selfest_c_list.append('a'+str(x))
    selfestd.columns = selfest_c_list
    
    
    #Opinion Lead
    traits2 = pd.read_csv('Pers traits2.csv', sep=';', header=0)
    ids2=traits2.filter(regex=("ID"))
    opiniond=traits2.filter(regex=("DI_opionlead_W021_D01_I01_DI_opionlead.*"))
    opiniond=pd.concat([ids2,opiniond],axis=1)
    #print(repr(len(opiniond.columns)) + ' opiniond')   
    
    opinion_c_list=[]
    opinion_c_list.append('ID')
    for x in range(1, len(opiniond.columns)):
        opinion_c_list.append('a'+str(x))              
    opiniond.columns = opinion_c_list
        
    # get all child of interest
    fin=allChildrenInClass()
    fin=fin[fin.cl==classID]
    # create two empty columns that will represent the openness and expressiveness of each child
    fin['expr'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
    fin['open'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
    
    childrenlist=fin['child'].tolist()
    

    numZeroExpr=0
    numZeroOpen=0
    problematicChildList=[]
    for childID in childrenlist:
        #define the expressiveness and openness
        ex=0
        op=0    
        numOp=0
        numEx=0

        # all questions mapped here
        # for reversed use (1 - mapS_I)
        # 4 questions for openness / 16 questions for expressiveness (1 reversed)
 
        if nutrid.at[nutrid.index[nutrid.ID==childID][0],'a4']!=' ':
            op=op+mapS_I(nutrid.at[nutrid.index[nutrid.ID==childID][0],'a4'])
            numOp=numOp+1
        
        if nutrid.at[nutrid.index[nutrid.ID==childID][0],'a7']!=' ':
            op=op+mapS_I(nutrid.at[nutrid.index[nutrid.ID==childID][0],'a7'])
            numOp=numOp+1
            
        if opiniond.at[opiniond.index[opiniond.ID==childID][0],'a1']!=' ':
            ex=ex+mapS_I(opiniond.at[opiniond.index[opiniond.ID==childID][0],'a1'])
            numEx=numEx+1
        
        if opiniond.at[opiniond.index[opiniond.ID==childID][0],'a2']!=' ':
            op=op+mapS_I(opiniond.at[opiniond.index[opiniond.ID==childID][0],'a2'])
            numOp=numOp+1
            
        if opiniond.at[opiniond.index[opiniond.ID==childID][0],'a3']!=' ':
            ex=ex+mapS_I(opiniond.at[opiniond.index[opiniond.ID==childID][0],'a3'])
            numEx=numEx+1
        
        if opiniond.at[opiniond.index[opiniond.ID==childID][0],'a4']!=' ':
            ex=ex+mapS_I(opiniond.at[opiniond.index[opiniond.ID==childID][0],'a4'])
            numEx=numEx+1
            
        if opiniond.at[opiniond.index[opiniond.ID==childID][0],'a5']!=' ':
            ex=ex+mapS_I(opiniond.at[opiniond.index[opiniond.ID==childID][0],'a5'])
            numEx=numEx+1  
            
        if needbeld.at[needbeld.index[needbeld.ID==childID][0],'a4']!=' ':
            ex=ex+mapS_I(needbeld.at[needbeld.index[needbeld.ID==childID][0],'a4'])
            numEx=numEx+1
            
        if needbeld.at[needbeld.index[needbeld.ID==childID][0],'a5']!=' ':
            ex=ex+mapS_I(needbeld.at[needbeld.index[needbeld.ID==childID][0],'a5'])
            numEx=numEx+1       
            
        if needbeld.at[needbeld.index[needbeld.ID==childID][0],'a7']!=' ':
            ex=ex+mapS_I(needbeld.at[needbeld.index[needbeld.ID==childID][0],'a7'])
            numEx=numEx+1
            
        if needbeld.at[needbeld.index[needbeld.ID==childID][0],'a9']!=' ':
            ex=ex+mapS_I(needbeld.at[needbeld.index[needbeld.ID==childID][0],'a9'])
            numEx=numEx+1  
            
        if socd.at[socd.index[socd.ID==childID][0],'a1']!=' ':
            op=op+mapS_I(socd.at[socd.index[socd.ID==childID][0],'a1'])
            numOp=numOp+1
        
        if socd.at[socd.index[socd.ID==childID][0],'a2']!=' ':
            ex=ex+mapS_I(socd.at[socd.index[socd.ID==childID][0],'a2'])
            numEx=numEx+1
            
        if socd.at[socd.index[socd.ID==childID][0],'a3']!=' ':
            ex=ex+mapS_I(socd.at[socd.index[socd.ID==childID][0],'a3'])
            numEx=numEx+1
        
        if socd.at[socd.index[socd.ID==childID][0],'a4']!=' ':
            ex=ex+mapS_I(socd.at[socd.index[socd.ID==childID][0],'a4'])
            numEx=numEx+1
            
        if socd.at[socd.index[socd.ID==childID][0],'a5']!=' ':
            ex=ex+mapS_I(socd.at[socd.index[socd.ID==childID][0],'a5'])
            numEx=numEx+1
            
        if selfestd.at[selfestd.index[selfestd.ID==childID][0],'a1']!=' ':
            ex=ex+mapS_I(selfestd.at[selfestd.index[selfestd.ID==childID][0],'a1'])
            numEx=numEx+1
            
        if selfestd.at[selfestd.index[selfestd.ID==childID][0],'a3']!=' ':
            ex=ex+mapS_I(selfestd.at[selfestd.index[selfestd.ID==childID][0],'a3'])
            numEx=numEx+1
        
        if selfestd.at[selfestd.index[selfestd.ID==childID][0],'a5']!=' ':
            ex=ex+(1-mapS_I(selfestd.at[selfestd.index[selfestd.ID==childID][0],'a5']))
            numEx=numEx+1
            
        if selfestd.at[selfestd.index[selfestd.ID==childID][0],'a10']!=' ':
            ex=ex+mapS_I(selfestd.at[selfestd.index[selfestd.ID==childID][0],'a10'])
            numEx=numEx+1
        
         
        # nodes that have not answered expressiveness/openness questions, will get the average of the graph as value
        if numEx!=0:    
            fin.at[childID,'expr']=ex/numEx
        else:
            fin.at[childID,'expr']=-1
            problematicChildList.append(childID)
            numZeroExpr=numZeroExpr+1
            
        if numOp!=0:    
            fin.at[childID,'open']=op/numOp
        else:
            fin.at[childID,'open']=-1
            problematicChildList.append(childID)
            numZeroOpen=numZeroOpen+1
            
        # kids that answered questions, but their result value is 0    
        if ex==0:
            problematicChildList.append(childID)
            #print('zero expr'+repr(childID))

        if op==0:
            problematicChildList.append(childID)
            #print('zero open'+repr(childID))
    
    # check if some of the kids have values 0, or -1
    # nodes that have answered, and have value 0 will get the min value bigger than 0
    avge=fin.expr[fin.expr >= 0].mean()
    avgo=fin.open[fin.open >= 0].mean()
    
    mine=fin.expr[fin.expr > 0].min()
    mino=fin.open[fin.open > 0].min()

    for c in problematicChildList:
        
        if fin.at[c,'expr']==-1:
            fin.at[c,'expr']=avge
        elif fin.at[c,'expr']==0:
            fin.at[c,'expr']=mine
            
        if fin.at[c,'open']==-1:
            fin.at[c,'open']=avgo
        elif fin.at[c,'open']==0:
            fin.at[c,'open']=mino
    
    # after this point all kids should have some value for exp and open.
    # now add the personal characteristics options
    
    # now deal with the self descriptive terms
    # Self Descriptive
    selfd=traits.filter(regex=("Gen_fill_Selfdescrip_W019_D03_I01_Selfdescrip.*"))
    selfd=pd.concat([ids,selfd],axis=1)
    selfd=selfd.fillna(0)
    
    lval=selfd.values.tolist()
    flist=[]

    for l in lval:
        if l[0] in childrenlist:
            if 0 in l:
                le=list(filter((0).__ne__, l))
            flist.append(le)
            
    # add the self-desc traits to the calculation        
    for childTraits in flist:
        nexpr,nopen=mapSelfDesc(listTraits=childTraits,oldEx=fin.at[childTraits[0],'expr'],oldOp=fin.at[childTraits[0],'open'])
        fin.at[childTraits[0],'expr']=nexpr if nexpr<=1 else 1
        fin.at[childTraits[0],'open']=nopen if nopen<=1 else 1
    
    return fin
if __name__ == "__main__":
    # execute only if run as a script
    graph_all = generate_network_PA(label='all')
    
    
# from script tuning.py    
    
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



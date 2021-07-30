### Step 1: Generate the Interventions results
#### For Gen Network and 15% target intervention
#### Also generate the gephi files and write the results to excel
#### We have all classes subgraphs in cGraphs and the intervention results in interventiondf

interventiondf,_,cGraphs,_=get_classes_intervention_results(network=['gen'],per=[15],generateGephiFiles=True, writeToExcel=True)


### Step 2 Generate all results that are useful for the analysis. This will create plots, docx, excel files ( can be skipped if wanted).  Individual Class results are saved under: Results/Class{ID}, while joint results are saved under Results/ClassesSummary 
#### The following methods are called:
   ##### get_interventions_differences -> output the 'effectiveness' of the interventions by looking at the differences in running the model with intervention VS nointervention. In addition for each intervention we show  PA  increase or decrease, after one year of running, in percentages. 
   ##### get_intervention_per_child_plots -> plots for PA changes over one year, for each children of each class.
   ##### get_intervention_model_comparison_plots -> plots comparing the effects of running the contagion VS diffusion model, per every intervention, network type and percentage. For each class in the network.
   ##### get_all_interventions_per_model_plots -> comparing the effects of all the interventions, for each model.
   ##### get_classes_intervention_comparison_plots -> comparing the effects of running a particular intervention at all the classes. We can see that running the same intervention results in different outcomes per class basis. 


intdiff=get_interventions_differences(class_dict=interventiondf,writeToExcel=True)
get_intervention_per_child_plots(classes_results=interventiondf, save_png=True, create_doc=True)
get_intervention_model_comparison_plots(classes_results=interventiondf, save_png=True, create_doc=True)
get_all_interventions_per_model_plots(classes_results=interventiondf, save_png=True, create_doc=True)
get_classes_intervention_comparison_plots(classes_results=interventiondf, save_png=True, create_doc=True,label=['gen'], percent=[15], model=['diffusion'])
_, classdf, childdf=get_df_class_children_topology_analysis(graphGen=cGraphs,network=['gen'],generateGephiFiles=True, generateExcel=True)





# # model validation per class level

# fitbit = pd.read_csv('fitbit.csv', sep=';', header=0)
# classes=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121,122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# fitbit = fitbit[fitbit['Class'].isin(classes)]
# empirical_class67= fitbit[fitbit['Class']==67.0]
# empirical_class67=empirical_class67[['Child_Bosse','Steps_ML_imp1','Date']]
# empirical_class67=empirical_class67[np.isfinite(empirical_class67['Steps_ML_imp1'])]
# # empirical_class67['Date'].apply(lambda x: transfrom(x))
# empirical_class67['Date']= empirical_class67['Date'].apply(lambda x: transfrom(x))
# empirical_class67=empirical_class67.sort_values(by=['Child_Bosse','Date'])
# empirical_class67=empirical_class67.pivot(index='Date', columns='Child_Bosse')
# empirical_class67=empirical_class67.fillna(0)
# empirical_class67=empirical_class67.mean(axis=1)
# empirical_class67=empirical_class67* 0.000153
# empirical_class67

# def transfrom(s):
#     fmt = '%Y-%m-%d'
#     dt = datetime.datetime.strptime(s, fmt)
#     return dt.timetuple().tm_yday-1

# # the items we want to remove in the intervention data
# list_i = [e for e in list(range(0, 365)) if e not in empirical_class67.index.values]
# contagion_simulation_noint=df_interventions[0]['contagion']['nointervention']['gen'][15].drop(list_i)
# contagion_simulation_noint=contagion_simulation_noint.mean(axis=1)
# contagion_simulation_noint
# # finally, plot the calculations
# contagion_simulation_noint.plot()
# empirical_class67.plot()

# sqrt(mean_absolute_error(empirical_class67, contagion_simulation_noint))

# # get empirical as it is
# emp=get_empirical(classes=[67])
# emp=emp[[4]]
# emp

# # mean of all kids per class
# mean_class67_contagion=df_interventions[0]['contagion']['nointervention']['gen'][15].T
# PA_sim = mean_class67_contagion[[364]]

# error = ((PA_sim[364]-emp[4])**2).sum().sum()/10
# error

# mean_absolute_error(emp, PA_sim)
# # sqrt(mean_absolute_error(emp, PA_sim))

# # filter the self - descriptive columns
# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# selfd=traits.filter(regex=("Gen_fill_Selfdescrip_W019_D03_I01_Selfdescrip.*"))
# final=pd.concat([ids,selfd],axis=1)
# final=final.fillna(0)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=final.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         if 0 in l:
#             le=list(filter((0).__ne__, l))

#         if len(le)==1:
#             countEmpty=countEmpty+1
#         elif 2 <= len(le) <= 5:
#             between=between+1
#         elif len(le)==6:

#             exact=exact+1
#         elif len(le)>6:

#             more=more+1
            

#         fin.at[l[0],'ans'] = len(le)-1 
#         flist.append(le)
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','total','empty','less','exact','more','percSelfDesc'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp),len(temp[temp.ans==0]),len(temp[(temp.ans>0) & (temp.ans<5)]),len(temp[temp.ans==5]),len(temp[temp.ans>5]),round((1-abs((len(temp[temp.ans==5])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# dfPersTraits=classPerc[classPerc['cl','percSelfDesc']]

# # looking at most common self-desc words
# traits=Counter(x for xs in flist for x in set(xs))
# traits=traits.most_common(70)
# traits


# # filter the Nutri columns
# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# nutrid=traits.filter(regex=("DI_NutriLoC_W019_D03_I01_LoC.*"))
# fnutri=pd.concat([ids,nutrid],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=fnutri.values.tolist()
# flist=[]
# datalist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
        
#         datalist.append(l)
        
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','eight','other','percNutriLoc'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==8]),len(temp[(temp.ans!=8) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==8])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# # dfPersTraits=dfPersTraits.join(classPerc['percNutriLoc'])

# #GEN_Selfesteem_W019_D04_I01_S
# # filter the SelfEsteem columns
# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# selfed=traits.filter(regex=("GEN_Selfesteem_W019_D04_I01_S.*"))
# selfed=pd.concat([ids,selfed],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=selfed.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','ten','other','percSelfEsteem'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==10]),len(temp[(temp.ans!=10) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==10])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# dfPersTraits=dfPersTraits.join(classPerc['percSelfEsteem'])

# #Gen_BriefFearNegEv_W019_D05_I01_BFNE
# # filter the SelfEsteem columns
# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# fearnegd=traits.filter(regex=("Gen_BriefFearNegEv_W019_D05_I01_BFNE.*"))
# fearnegd=pd.concat([ids,fearnegd],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=fearnegd.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','twelwe','other','percNegativeEval'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==12]),len(temp[(temp.ans!=12) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==12])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# dfPersTraits=dfPersTraits.join(classPerc['percNegativeEval'])

# # Gen_Need2belong_W019_D02_I01_NtB

# # filter the SelfEsteem columns
# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# needbeld=traits.filter(regex=("Gen_Need2belong_W019_D02_I01_NtB.*"))
# needbeld=pd.concat([ids,needbeld],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=needbeld.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','ten','other','percNeedBelong'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==10]),len(temp[(temp.ans!=10) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==10])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)
# dfPersTraits=dfPersTraits.join(classPerc['percNeedBelong'])

# # GEN_prosocial_W019_D06_I01_GEN_prosocial

# traits = pd.read_csv('Pers traits.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# socd=traits.filter(regex=("GEN_prosocial_W019_D06_I01_GEN_prosocial.*"))
# socd=pd.concat([ids,socd],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=socd.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','five','other','percProSocial'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==5]),len(temp[(temp.ans!=5) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==5])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# dfPersTraits=dfPersTraits.join(classPerc['percProSocial'])

# # DI_opionlead_W021_D01_I01_DI_opionlead

# traits = pd.read_csv('Pers traits2.csv', sep=';', header=0)
# ids=traits.filter(regex=("ID"))
# opiniond=traits.filter(regex=("DI_opionlead_W021_D01_I01_DI_opionlead.*"))
# opiniond=pd.concat([ids,opiniond],axis=1)

# # take the relevant children from selected classes
# fin=allChildrenInClass()
# fin['ans'] = pd.Series(np.zeros((len(fin.index))), index=fin.index)
# clist=fin['child'].tolist()

# lval=opiniond.values.tolist()
# flist=[]
# countEmpty=0
# between=0
# exact=0
# more=0
# for l in lval:
#     if l[0] in clist:
#         le=[]
#         if ' ' in l:
#             le=list(filter((' ').__ne__, l))
#         else:
#             le=l
        
#         flist.append(le)
#         fin.at[l[0],'ans'] = len(le)-1 
    
# wantedClass=[67, 71, 72, 74, 77, 78, 79, 81, 83, 86, 100, 101, 103, 121, 122, 125, 126, 127, 129, 130, 131, 133, 135, 136, 138, 139]
# classPerc = pd.DataFrame(columns=['cl','empty','six','other','percOpinionLead'])
# for c in wantedClass:
#     temp=fin[fin.cl==c]
#     classPerc.loc[len(classPerc)] = [c,len(temp[temp.ans==0]),len(temp[temp.ans==6]),len(temp[(temp.ans!=6) & (temp.ans!=0)]),round((1-abs((len(temp[temp.ans==6])-len(temp))/len(temp)))*100,2)]
    
# print(classPerc)

# #dfPersTraits=dfPersTraits.join(classPerc['percOpinionLead'])

# df_to_excel(dfPersTraits,'open_express.xlsx','open_exp')
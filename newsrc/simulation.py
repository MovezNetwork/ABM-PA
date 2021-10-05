import networkx as nx
import json
import numpy as np
import os
import pandas as pd
import time
from docx import Document 
from xlwt import Workbook
import collections
from docx.shared import Inches
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns
from itertools import chain

import newsrc.population as p
import newsrc.model as m


class Simulation:
    def __init__(self, **args):
        self.input_args = self.load_input_args()
        self.population = p.Population('MyMovez school classes', self.input_args)
        self.model = m.DiffusionModel('Gabrianelli Diffusion Model', self.input_args)


    def load_input_args(self):
        try:
            input_args = json.loads(open('../input/simulation.json').read())
        except Exception as ex:
            print('simulation.json does not exist!')
            print(ex)
            
        return input_args
    
    
    def start(self,time):
        '''
        TODO: call to methods for simulation should go in here. All relevant methods should be knit together here
        '''
        pass
        

    '''
    TODO: should we include a "reset"(= clean up some objects; so you can restart a run without running init again) & "stop" (= clean up all objects) method
    '''
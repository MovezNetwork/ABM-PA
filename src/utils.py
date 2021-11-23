'''
Utility methods
'''

import json
import pandas as pd


def load_input_args(file):
    '''
    Read json input file
    '''
    try:
        input_args = json.loads(open(file).read())
    except Exception as ex:
        print(file, 'does not exist!')
        print(ex)

    return input_args


def get_PA_dictionary(graph):
    '''
    Convert a graph to dictionary
    '''

    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']

    return pd.DataFrame(PA_dict)


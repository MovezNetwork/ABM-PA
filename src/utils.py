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
    Getting a PA dictionary of the model's history PAL values after simulation.

    Args:
        graph (NetworkX graph): graph population after finishing the simulation

    Returns:
        dictionary: PAL history of the simulation per days
    '''

    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']

    return pd.DataFrame(PA_dict)


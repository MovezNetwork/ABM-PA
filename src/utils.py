'''
Utility methods
'''

import json
import pandas as pd
import numpy as np


def load_input_args(file):
    '''
    Read json input file
    Args:
        file (string): json file
    '''
    try:
        input_args = json.loads(open(file).read())
    except Exception as ex:
        print(file, 'does not exist!')
        print(ex)

    return input_args


def get_PA_dictionary(graph):
    '''
    Gets PA dictionary of the model's history PAL values after simulation.
    Args:
        graph (Graph): graph population after finishing the simulation
    Returns:
        dictionary: PAL history of the simulation per days
    '''

    results_dict = dict(graph.nodes(data=True))
    PA_dict = {}
    for k, v in results_dict.items():
        PA_dict[k] = results_dict[k]['PA_hist']

    return pd.DataFrame(PA_dict)



def get_empirical_data(file, classes):
    '''
    Reads empirical data from file

    Args:
        file (string): data file name
        classes (array): list of class ids

    Returns:
        dataframe with physical activity data (steps/10,000) per child and wave.
    '''

    df_pal = pd.read_csv(file, sep=';', header=0, encoding='latin-1')
    df_pal = df_pal[df_pal['Class'].isin(classes)]

    df_pal = df_pal.groupby(['Class', 'Wave']).mean()['Steps'].reset_index()

    # normalize the number of steps: divided by 10,000
    df_pal.Steps = df_pal.Steps * 0.0001
    df_pal = df_pal.pivot(index='Class', columns='Wave')['Steps']

    return df_pal


def fix_float64(orig_dict):
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



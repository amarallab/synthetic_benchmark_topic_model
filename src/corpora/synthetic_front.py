# <<< Add the src path to sys.path
import sys
import os

containing_folder = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(containing_folder, os.pardir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# >>>

from corpora.pp_single_stopword import synthetic_single_stopword_wrapper
from corpora.dirichlet import synthetic_dirichlet_wrapper


def synthetic_corpora_front(dict_input):
    '''
    The front interface for generating synthetic benchmark corpora.
    In this work, we can generate two major types of synthetic benchmark corpora:
    1. synthetic benchmark corpora based on Dirichlet distribution;
    2. synthetic benchmark corpora based on adjusting the degree of structure parameters.
    '''

    corpus_type = dict_input['corpus_type']

    if corpus_type == 'dirichlet':
        dict_output = synthetic_dirichlet_wrapper(dict_input)

    elif corpus_type == 'single_stop':
        dict_output = synthetic_single_stopword_wrapper(dict_input)

    else:
        print('Check the input dictionary! It does not match all available options!')

    return dict_output

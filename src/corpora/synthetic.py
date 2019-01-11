'''
Collection of functions related to drawing synthetic corpora.
'''

# <<< Add the src path to sys.path
import sys
import os

containing_folder = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(containing_folder, os.pardir))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# >>>


import numpy as np
from collections import Counter
import random


from common.convert_states import state_nwjd, nwd_to_texts


def draw_dwz_from_ptd_pwt(p_td, p_wt, m, burstiness=None):
    '''
    Given the  p(word | topic)- and p(topic | doc)- conditional probability matrices draw the labeled dwz-state
    In: - p_td, cond prob p(word | topic), array KxD
        - p_wt, cond prob p(topic | doc), array VxK
        - m, textlength
    Out:
        - state_dwz, the labeled corpus as list  [(doc-id,word-id,topic-id)]  with len= number of tokens in corpus
        - p_w_td, p(word | topic) for a specific document, i.e., p(word | topic, document); if burstiness is None, p(word | topic, doc1) ==  p(word | topic, doc2)
    '''

    state_dwz = []
    D_ = len(p_td[0, :])
    K_ = len(p_td[:, 0])
    V_ = len(p_wt[:, 0])

    p_w_td = np.zeros((D_, V_, K_))

    for i_d in range(D_):
        state_dwz_tmp = []  # state for this document
        p_t_d_tmp = p_td[:, i_d]

        # number of tokens for each topic
        list_t_tmp = list(np.random.choice(K_, size=m, replace=True, p=p_t_d_tmp))
        c_jd = Counter(list_t_tmp)

        for t in range(K_):
            p_w_t_tmp = p_wt[:, t]
            p_w_td[i_d, :, t] = p_w_t_tmp

        list_w_tmp = []
        for t, n_t in c_jd.items():
            p_w_t_tmp = p_wt[:, t]

            # # BURSTINESS !!! draw dirichlet - default
            if burstiness is not None:
                p_w_t_tmp = draw_dirichlet(burstiness * p_w_t_tmp * V_)
            p_w_td[i_d, :, t] = p_w_t_tmp
            # #
            list_ = list(np.random.choice(V_, size=n_t, replace=True, p=p_w_t_tmp))
            list_w_tmp += list_
            c_wd_tmp = Counter(list_)
            for w_, n_w_ in c_wd_tmp.items():
                state_dwz_tmp += [(i_d, w_, t)] * n_w_
        random.shuffle(state_dwz_tmp)
        state_dwz += state_dwz_tmp
    return state_dwz, p_w_td


def get_pw_pt(V, dist='uni', gamma=1.0):
    '''
    Define word frequency-distribution over a vocabulary of V terms.
    IN:
    - V, int: number of word-types
    - dist, str: type of distribution (default: uniform)
        - 'uni': p_w = 1/V
        - 'zipf': p_w(r) = r**(-1) r=1,...,V
        - if something else than that is provided, you will get a warning and the uniform case
    OUT:
    - arr_pw, arr of floats: len=V
    '''

    if dist == 'uni':
        arr_pw = 1.0 / V * np.ones(V)
    elif dist == 'zipf':
        arr_pw = (np.arange(V) + 1) ** (-gamma)
        arr_pw /= np.sum(arr_pw)
    else:
        print('Invalid distribution type. You will get the uniform distribution')
        arr_pw = 1.0 / V * np.ones(V)
    return arr_pw

# def get_pt(K, dist = 'uni'):
#     '''
#     Define the distribution over topics (i.e. the 'size' of the topics) over K topics.
#     IN:
#     - K, int: number of topics
#     - dist, str: type of distribution (default: uniform)
#         - 'uni': p_j = 1/K
#         - 'zipf': p_j = j**(-1) j=1,...,K
#         - if something else than that is provided, you will get a warning and the uniform case
#     OUT:
#     - arr_j, arr of floats: len=K
#     '''
#     if dist=='uni':
#         arr_pj = 1.0/K*np.ones(K)
#     elif dist=='zipf':
#         gamma = 1.0
#         arr_pj = (np.arange(K)+1)**(-gamma)
#         arr_pj /= np.sum(arr_pj)
#     else:
#         print('Invalid distribution type. You will get the uniform p_j')
#         arr_pj = 1.0/K*np.ones(K)
#     return arr_pj


def make_dict_corpus_for_inference(dict_output_corpus):
    '''
    Take output from topicmodel_synthetic_front [generating the synthetic corpus]
    and put in the form that it can be used by topicmodel_inference_front [inferring the synthetic corpus]
    IN:
    -dict, contains 'state_dwz','p_wt','p_td'
    OUT:
    - dict, contains  'texts_list_shuffle', 'state_dwz_shuffle'
    '''
    state_dwz = dict_output_corpus['state_dwz']
    p_wt = dict_output_corpus['p_wt']
    p_td = dict_output_corpus['p_td']
    V, K = np.shape(p_wt)
    K, D = np.shape(p_td)
    # ## convert state into corpus
    n_wd, n_wj, n_jd = state_nwjd(state_dwz, D, V, K)
    texts = nwd_to_texts(n_wd.astype('int'))
    dict_corpus_tmp = {
        'texts_list_shuffle': texts,
        'state_dwz_shuffle': state_dwz}
    return dict_corpus_tmp


def make_dict_input_corpus(corpustype, K, D, m, V, par_structure, dist_w='uni', dist_z='uni', p_n=None):
    '''
    Given the corpustype we set up the dictionary for the corpus-wrapper.
    IN:
        - corpustype, str
        - K, int; number of topics
        - D, int; number of document
        - m, int; length of texts
        - V, int; number of different words
        - par_structure, list/float; the vlaues of the structure parameters in each model
            - if list, the structure parameters are filled succesively
            - if float, all structure parameters get the same value
        - dist_w, int; global word distribution (default 'uni'); options: 'uni', 'zipf'
        - dist_z, int; global topic size distribution (default 'uni'); options: 'uni', 'zipf'
    OUT:
        - dict_input_corpus, dict
            {'D':D, 'V':V, 'm':m, 'K': K ## size-parameters
              'dist_w':dist_w, 'dist_z': dist_z ## external parameters (default: uni)

              structure parameters are set for each model from par_structure (list or float)
            }
    '''

    dict_input_corpus_template = {
        'corpustype': corpustype,
        'K': K,
        'D': D,
        'm': m,
        'V': V,
        'dist_z': dist_z,
        'dist_w': dist_w, }

    dict_input_corpus = dict(dict_input_corpus_template)

    # #
    # ## pp-single
    # #
    if corpustype == 'pp_single':
        if isinstance(par_structure, list):
            if len(par_structure) == 1:
                c_w = par_structure[0]
                c_d = 1.0 * c_w
            elif len(par_structure) > 1:
                c_w = par_structure[0]
                c_d = par_structure[1]
            else:
                c_w, c_d = 1.0, 1.0
        elif isinstance(par_structure, (int, float)):
            c_w = 1.0 * par_structure
            c_d = 1.0 * par_structure
        dict_input_corpus['c_w'] = c_w
        dict_input_corpus['c_d'] = c_d
        # # distribution of docs belonging to several topics
        # if not supplied a doc is only assigned to one topic
        if p_n is None:
            p_n = np.array([1.0])
        dict_input_corpus['p_n'] = p_n

    ##################################
    ##
    # ## dirichlet
    ##
    elif corpustype == 'dirichlet':
        if isinstance(par_structure, list):
            if len(par_structure) == 1:
                alpha = par_structure[0]
                beta = 1.0 * alpha
            elif len(par_structure) > 1:
                alpha = par_structure[0]
                beta = par_structure[1]
            else:
                alpha, beta = 1.0, 1.0
        elif isinstance(par_structure, (int, float)):
            alpha = 1.0 * par_structure
            beta = 1.0 * par_structure
        dict_input_corpus['alpha'] = alpha
        dict_input_corpus['beta'] = beta
    else:
        dict_input_corpus = {}
        print('SPECIFY valid corpustype')
    return dict_input_corpus


def draw_dirichlet(vec_alpha, seed=None):
    '''
    Draw a sample vec_theta from a Dirichlet-distribution with hyperparameter vec_alpha.
    IN:
    -vec-alpha: len=dimensionality of hyperparameter
    OUT:
    -vec_theta: len=dimensionality of output-vector (same as vec_alpha), with sum(vec_theta)=1

    There is a known bug in np.random.dirichlet for small values of alpha
    https://github.com/numpy/numpy/issues/5851
    There are two possible outcomes:
    - ZeroDivisionError
    - np.nan
    The reason for this is that the random numbers are drawn from Beta-random numbers.
    Thus when the alpha-values are too small we divide by zero (no weight in any of the dimensions).
    However, we can just select one of the dimensions as having weight 1;
    We therefore just draw a random index (of course weighted by prob vec_alpha/sum(vec_alpha))
    '''
    if np.max(vec_alpha) < 0.01:
        # if concentration parameter too small we will just get one entry with all the mass
        # pick one index (weighted by alpha)
        vec_p = vec_alpha / np.sum(vec_alpha)
        np.random.seed(seed=seed)
        ind_theta = np.random.choice(len(vec_alpha), p=vec_p)
        vec_theta = np.zeros(len(vec_alpha))
        vec_theta[ind_theta] = 1.0
    else:
        np.random.seed(seed=seed)
        vec_theta = np.random.dirichlet(vec_alpha)
    return vec_theta


def sort_synthetic_corpora_output(dict_input):
    '''
    Order the p_wt, p_td, n_wd in approx block-diagonal form.
    IN:
    - the output dictionary from the corpora-wrapper
    OUT:
    - p_td_sort
    - p_wt_sort
    - n_wd_sort_sort
    '''

    # Get sorted index
    document_topic_assign_list = dict_input['document_topic_assign_list']
    word_topic_assign_list = dict_input['word_topic_assign_list']

    sort_index_document = np.argsort(document_topic_assign_list)
    sort_index_word = np.argsort(word_topic_assign_list)

    # Get the data need to plot
    p_td = dict_input['p_td']
    p_wt = dict_input['p_wt']

    n_wd = dict_input['n_wd']
    n_wj = dict_input['n_wj']
    n_jd = dict_input['n_jd']
    # Sort the data
    p_td_sorted = p_td[:, sort_index_document]
    p_wt_sorted = p_wt[sort_index_word]

    n_wd_sorted = n_wd[sort_index_word][:, sort_index_document]
    n_wj_sorted = n_wj[sort_index_word]
    n_jd_sorted = n_jd[:, sort_index_document]

    return p_td_sorted, p_wt_sorted, n_wd_sorted, n_wj_sorted, n_jd_sorted


def formulate_input_for_topicModel(
        dict_output_corpus, topic_model,
        K_pp=None, V_pp=None,
        input_k=None,
        set_alpha=None, set_beta=None,
        iterations=None, gamma_threshold=None,
        flag_holdout=0, trained_ratio=0.9,):
    '''
    Formulate the input dictionary for each topic model algorithm: ldavb, ldags, hdp, tm, sbm

    Input:
    dict_output_corpus =
    {
    'texts_list_shuffle':       ## the corpus
    'state_dwz_shuffle':        ## list of tuple, do not exist for real-world corpus
    }
    topic_model: str, the name of topic model
    K_pp: int, real/true number of topics
    V_pp: int, number of word types in the vocabulary
    input_k:    int, input number of topics for models (e.g., ldavb, ldags) which needs is as a parameter,
                if not given, set input_k = K_pp

    Output:
    dict_input_topicmodel=
    {
    'topic_model':   ## the name of topic model
    'texts':
    ....
    }   ## the output is model-specific
    '''

    if input_k is None:
        input_k = K_pp

    texts_list_shuffle = dict_output_corpus['texts']
    state_dwz_shuffle = dict_output_corpus.get('state_dwz', None)

    dict_input_topicmodel = {}
    dict_input_topicmodel['topic_model'] = topic_model

    if topic_model == 'ldavb':
        dict_input_topicmodel['texts'] = texts_list_shuffle
        dict_input_topicmodel['input_k'] = input_k

        # Set the hyperparameter, the iteration number for the algorithm, and the stop criteria
        if set_alpha is not None:
            dict_input_topicmodel['set_alpha'] = set_alpha
        if set_beta is not None:
            dict_input_topicmodel['set_beta'] = set_beta
        if iterations is not None:
            dict_input_topicmodel['iterations'] = iterations
        if gamma_threshold is not None:
            dict_input_topicmodel['gamma_threshold'] = gamma_threshold

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp
            dict_input_topicmodel['input_v'] = V_pp

        if flag_holdout != 0:
            dict_input_topicmodel['flag_holdout'] = flag_holdout
            dict_input_topicmodel['trained_ratio'] = trained_ratio

    if topic_model == 'ldags':
        dict_input_topicmodel['texts'] = texts_list_shuffle
        dict_input_topicmodel['input_k'] = input_k

        # Set the hyperparameter and the iteration number for the algorithm
        if set_alpha is not None:
            dict_input_topicmodel['set_alpha'] = set_alpha
        if set_beta is not None:
            dict_input_topicmodel['set_beta'] = set_beta
        if iterations is not None:
            dict_input_topicmodel['iterations'] = iterations

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

        if flag_holdout != 0:
            dict_input_topicmodel['flag_holdout'] = flag_holdout
            dict_input_topicmodel['trained_ratio'] = trained_ratio

    if topic_model == 'hdp':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    if topic_model == 'tm':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    if topic_model == 'sbm':
        dict_input_topicmodel['texts'] = texts_list_shuffle

        if state_dwz_shuffle is not None:
            dict_input_topicmodel['state_dwz_true'] = state_dwz_shuffle
            dict_input_topicmodel['k_true'] = K_pp

    return dict_input_topicmodel

'''
Collection of functions to drawing synthetic corpora based on Dirichlet distribution.
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


from corpora.synthetic import draw_dwz_from_ptd_pwt, get_pw_pt, draw_dirichlet
from corpora.structure import deltaI_from_nwd
from common.convert_states import state_nwjd, nwd_to_texts


def make_pwt_ptd_dirichlet(vec_alpha, vec_beta, D, seed=None):
    '''From Dirichlet hyperparameters, generate the conditional probabilities
    - p(t|d)
    - p(w|t)
    IN:
    - vec_alpha: hyperparameter for topic-doc distribution, len=K...number of topics
    - vec_beta: hyperparameter for word-topic distribution, len=V...number of word-types
    - D: number of documents, int
    OUT:
    - p_td:  p(t|d), shape=KxD
    - p_wt:  p(w|t), shape=VxK
    '''
    K = len(vec_alpha)
    V = len(vec_beta)

    # get seed_doc_list and seed_topic_list
    if seed is not None:
        np.random.seed(seed=seed)
        seed_doc_list = np.random.choice(D * D, size=D, replace=False)
        np.random.seed(seed=seed + 100)
        seed_topic_list = np.random.choice(K * K, size=K, replace=False)

    # # p_t_d from samples of alpha_vec
    p_td = np.zeros((K, D))
    for i_d in range(D):
        if seed is None:
            seed_doc = None
        else:
            seed_doc = seed_doc_list[i_d]
        p_td_tmp = draw_dirichlet(vec_alpha, seed=seed_doc)
        p_td[:, i_d] = 1.0 * p_td_tmp
    # # p_w_t from samples of beta_vec
    p_wt = np.zeros((V, K))
    for i_K in range(K):
        if seed is None:
            seed_topic = None
        else:
            seed_topic = seed_topic_list[i_K]
        p_wt_tmp = draw_dirichlet(vec_beta, seed=seed_topic)
        p_wt[:, i_K] = 1.0 * p_wt_tmp

    return p_td, p_wt


def make_hyper_vec(alpha, vec_p):
    '''make the hyperparameter vector in our notation.
    The resulting dirichlet-hyperparameter is given by:
    vec{alpha} = (alpha_1,alpha_2,...,alpha_S) with S = len(vec_p) with alpha_i = alpha*S*p_i.
    If we draw random variables vec{x}, the marginal expectation value is <x_i> = p_i

    IN:
    - alpha, float
    - vec_p = (p_i) with i=1,..,S
    OUT:
    - vec_alpha = (alpha_i) with i =1,...,S
    '''
    return alpha * len(vec_p) * vec_p


def synthetic_dirichlet_wrapper(dict_input, if_info=True):
    '''
    Wrapper for drawing synthetic corpus: DIRICHLET

    Input:
        dict_input = {
            ## choose topic model
            'corpustype': 'dirichlet', str

            ## corpus-size parameters
            , 'D':D, int  ;#of docs
              'V':V, int  ;#of word-types
              'm':m, int  ;textlength
              'K':K, int  ;#of topics

            ## structure parameters
              'alpha':alpha, float   ; concentration parameter for topic-doc distribution p_td
              'beta':beta, float   ; concentration parameter for word-topic distribution p_wt
            ## external contraints parameters
            'dist_w': dist_w, str ; global distribution of word-frequencies
                                    'uni','zipf'
            'dist_z': dist_z, str ; global distribution of topic-frequencies
                                    'uni','zipf'
        }

    Output:
        dict_output = {
            'state_dwz':[(d,w,z)]; list of 3-tuples for each drawn token (doc,word-type,topic-label)
            list_t_d_true:[ t_d]; topic-membership of each document
            'p_wt':p_wt; word-topic dist, arr.shape=VxK
            'p_td':p_td; topic-doc dist, arr.shape=KxD
        }
    '''

    # # get the size-parameters

    V = dict_input['V']
    K = dict_input['K']

    # # external contraints parameters
    dist_w = dict_input['dist_w']

    # # get the structure parameters
    alpha = dict_input['alpha']

    # # Parameters for document
    D = dict_input['D']
    m = dict_input['m']

    # # optional parameter
    beta = dict_input.get('beta', None)
    dist_t = dict_input.get('dist_t', None)
    seed = dict_input.get('seed', None)
    burstiness = dict_input.get('burstiness', None)

    #################################

    dict_output = synthetic_dirichlet_terminal(V, K, dist_w, D, m, alpha, beta, dist_t=dist_t, seed=seed, burstiness=burstiness, if_info=if_info)

    return dict_output


def synthetic_dirichlet_terminal(V, K, dist_w, D, m, alpha, beta=None, dist_t=None, seed=None, burstiness=None, if_info=True):

    if dist_t is None:
        dist_t = dist_w
    if beta is None:
        beta = 1.0 * alpha

    # # global distribution of topic-size
    p_t = get_pw_pt(K, dist=dist_t)
    # # global distribution of word frequencies
    p_w = get_pw_pt(V, dist=dist_w)

    # # get vector-hyperparameters
    vec_alpha = make_hyper_vec(alpha, p_t)
    vec_beta = make_hyper_vec(beta, p_w)

    # # create the mixture-matrices p_wt (word-topic) and p_td (topic-doc)
    p_td, p_wt = make_pwt_ptd_dirichlet(vec_alpha, vec_beta, D, seed=seed)

    # # draw the dwz-state
    state_dwz, p_w_td = draw_dwz_from_ptd_pwt(p_td, p_wt, m, burstiness=burstiness)
    n_wd, n_wj, n_jd = state_nwjd(state_dwz, D, V, K)
    texts = nwd_to_texts(n_wd)
    # # infer the topic-membership of each doc:
    # # choose topic with largest contribution from p(t|d)
    list_t_d_true = np.argmax(p_td, axis=0)

    # # empirical p_t and p_w; otherwise p_tw is not normalized
    p_t_emp = 1.0 / D * np.sum(p_td, axis=1)
    p_w_emp = np.sum(p_wt * p_t_emp)

    # # infer the topic-membership of each word:
    # # choose topic with largest contribution from p(t|w) = p(w|t)*p(t)/p(w)
    # p_tw = (p_wt*(p_w[:,np.newaxis]/p_t)).T
    p_tw = (p_wt.T * (p_t_emp[:, np.newaxis] / p_w_emp))
    list_t_w_true = np.argmax(p_tw, axis=0)

    # # Get the structure
    if if_info:
        DeltaI, I_alpha = deltaI_from_nwd(n_wd)
    else:
        DeltaI = 0
        I_alpha = 0

    dict_out = {}
    dict_out['p_w'] = p_w
    # dict_out['V_t'] =V_t
    dict_out['word_topic_assign_list'] = list_t_w_true
    dict_out['p_t'] = p_t
    dict_out['p_wt'] = p_wt
    dict_out['document_topic_assign_list'] = list_t_d_true
    dict_out['p_td'] = p_td
    dict_out['state_dwz'] = state_dwz
    dict_out['n_wd'] = n_wd
    dict_out['n_wj'] = n_wj
    dict_out['n_jd'] = n_jd
    dict_out['texts'] = texts
    dict_out['p_tw'] = p_tw

    dict_out['DeltaI'] = DeltaI
    dict_out['I_alpha'] = I_alpha

    return dict_out

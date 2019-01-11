# from corpora.pp_single_stopword import
# # #

# System package
import numpy as np

# Private package
from corpora.synthetic import draw_dwz_from_ptd_pwt, get_pw_pt
from common.convert_states import state_nwjd, nwd_to_texts


def synthetic_single_stopword_wrapper(dict_input={}, if_info=0):
    '''
    Input:
    dict_input = {
    # Important parameter
    V: number of words
    K: number of topics
    p_s: percentage of stopwords
    c_w: degree of structure for word-topic distribution
    c_t: degree of structure for topic-document distribution

    }

    '''

    # Important parameter
    # # get the size-parameters
    V = dict_input.get('V', 1000)
    K = dict_input.get('K', 5)

    # # parameter for stopwords
    p_s = dict_input.get('p_s', 0)

    # # get the degree of structure parameters
    c_w = dict_input.get('c_w', 1)

    # # Parameters for document
    D = dict_input.get('D', 100)
    m = dict_input.get('m', 100)

    # optional parameter
    dist_stop = dict_input.get('dist_stop', 'uni')
    dist_w = dict_input.get('dist_w', 'uni')

    dist_t = dict_input.get('dist_t', None)
    c_t = dict_input.get('c_t', None)
    seed = dict_input.get('seed', None)
    burstiness = dict_input.get('burstiness', None)

    #################################
    dict_output = synthetic_single_stopword_terminal(
        V=V, K=K, D=D, m=m,
        dist_w=dist_w, dist_t=dist_t,
        dist_stop=dist_stop, p_s=p_s,
        c_w=c_w, c_t=c_t, seed=seed,
        burstiness=burstiness,
        if_info=if_info)

    return dict_output


def synthetic_single_stopword_terminal(
        V=1000, K=5, D=100, m=100,
        dist_w='uni', dist_t=None,
        dist_stop='uni', p_s=0,
        c_w=1, c_t=None, seed=None,
        burstiness=None,
        if_info=0):
    '''
    Output:
    p_w_td: p(w|t,d), in general, for each d, p(w|t,d) = p(w|t); however, for the burstiness case, p(w|t,d) is different for each document
    '''

    if dist_t is None:
        dist_t = dist_w

    if c_t is None:
        c_t = c_w

    # Get global word distribution
    # p_w = get_global_word_distribution_pw(V, marker_pw )
    p_w = get_pw_pt(V, dist=dist_w)

    # Get stopword distribution
    # stop_distrib = get_global_word_distribution_pw(V , marker_stop )
    stop_distrib = get_pw_pt(V, dist=dist_stop)

    # Choose stopword list
    num_stopword = int(V * p_s)
    np.random.seed(seed=seed)
    stopword_list = np.random.choice(V, size=num_stopword, replace=False, p=stop_distrib)

    # Get the number of word type in each topic
    num_nonstop = V - num_stopword
    V_t = get_vt_from_nonstop(K, num_nonstop, dist_t)  # V_t is the topic size for each topic, i.e., the number of useful non-stopwords in each topic

    # Get the topic assignment for each words: both stopwords and non-stopwords.
    # For stopwords, assign a very large number as their topic id.
    word_topic_assign_list = get_word_topic_assign_list(V_t, stopword_list, seed=seed)

    # Get topic distribution p_t
    p_t = get_topic_distribution_p_t(K, p_w, word_topic_assign_list)

    # Get word-topic distribution
    p_wt = get_word_topic_distribution_p_wt(K, p_w, p_t, word_topic_assign_list, c_w)

    # Get the topic assignment for each document
    document_topic_assign_list = np.random.choice(K, size=D, replace=True, p=p_t)

    # Get topic-document distribution
    p_td = get_topic_doc_distribution_ptd(K, p_t, c_t, document_topic_assign_list)

    # Get the synthetic corpus
    state_dwz, p_w_td = draw_dwz_from_ptd_pwt(p_td, p_wt, m, burstiness=burstiness)
    n_wd, n_wj, n_jd = state_nwjd(state_dwz, D, V, K)
    texts = nwd_to_texts(n_wd)

    # Get the output dictionarr

    dict_out = {}
    dict_out['p_w'] = p_w
    dict_out['V_t'] = V_t
    dict_out['word_topic_assign_list'] = word_topic_assign_list
    dict_out['p_t'] = p_t
    dict_out['p_wt'] = p_wt
    dict_out['p_w_td'] = p_w_td
    dict_out['document_topic_assign_list'] = document_topic_assign_list
    dict_out['p_td'] = p_td
    dict_out['state_dwz'] = state_dwz
    dict_out['n_wd'] = n_wd
    dict_out['n_wj'] = n_wj
    dict_out['n_jd'] = n_jd
    dict_out['texts'] = texts

    # Get the structure
    if if_info:
        DeltaI, I_alpha = deltaI_from_nwd(n_wd)
        dict_out['DeltaI'] = DeltaI
        dict_out['I_alpha'] = I_alpha

    return dict_out


def get_vt_from_nonstop(K, num_nonstop, dist_t):

    if num_nonstop < K:
        print('Number of topics is larger than the number of non-stop words ')

    if dist_t == 'zipf':

        # non-flat
        # if K<10: 80-20 rule here: 20% of topics get 80% of word-types
        if K < 10:
            ind1 = np.max([1, int(0.2 * K)])
            V_t = np.zeros(K)
            p1 = 0.8
            V_t[:ind1] = int(p1 / float(ind1) * num_nonstop)
            V_t[ind1:] = ((1.0 - p1) * num_nonstop * np.ones(K - ind1) / (K - ind1))
            V_t = V_t.astype('int')
            delta_V = num_nonstop - np.sum(V_t)
            V_t[:delta_V] += 1

        # if K>=10: the distribution of topic sizes is a powerlaw
        else:
            V_t = (np.arange(K) + 1)**(-1.0)
            V_t = num_nonstop * V_t / np.sum(V_t)
            V_t = V_t.astype('int')
            delta_V = num_nonstop - np.sum(V_t)
            V_t[K - delta_V:] += 1

    # flat-case
    elif dist_t == 'uni':

        V_t = (np.ones(K) * num_nonstop / K).astype('int')
        # for special case
        delta_V = num_nonstop - np.sum(V_t)
        V_t[K - delta_V:] += 1

    else:
        print('Invalid distribution type. You will get the uniform distribution')
        V_t = (np.ones(K) * num_nonstop / K).astype('int')
        # for special case
        delta_V = num_nonstop - np.sum(V_t)
        V_t[K - delta_V:] += 1

    # # Test if there is zero word for one topic
    V_t_nonzero = [ele for ele in V_t if ele > 0]
    if len(V_t_nonzero) != K:
        print('There is no word for at least one topic!')
        print('Please increase the number of nonstop words')
        return 'Error'

    return V_t


def get_word_topic_assign_list(V_t, stopword_list, seed=None):

    np.random.seed(seed=seed)

    V = sum(V_t) + len(stopword_list)
    K = len(V_t)

    # Get shuffled nonstop_topic_list
    nonstop_topic_list = []
    for i in range(len(V_t)):
        nonstop_topic_list += [i] * V_t[i]
    np.random.shuffle(nonstop_topic_list)

    # Get word_topic_assign_list
    word_topic_assign_list = []
    nonstop_id = 0
    for i in range(V):
        if i in stopword_list:
            word_topic_assign_list += [K + 100]
        else:
            word_topic_assign_list += [nonstop_topic_list[nonstop_id]]
            nonstop_id += 1

    return word_topic_assign_list


def get_topic_distribution_p_t(K, p_w, word_topic_assign_list):

    p_t = np.zeros(K)
    for t in range(K):
        p_t[t] = float(np.sum(p_w * (np.array(word_topic_assign_list) == t)))
    p_t /= sum(p_t)

    return p_t


def get_word_topic_distribution_p_wt(K, p_w, p_t, word_topic_assign_list, c_w):
    V = len(word_topic_assign_list)
    p_wt = np.zeros((V, K))

    for t in range(K):
        for w_id, w_t in enumerate(word_topic_assign_list):

            # For non-stopwords, and their topic ids are t
            if w_t == t:
                p_wt[w_id, t] = c_w * p_w[w_id] / p_t[t] + (1 - c_w) * p_w[w_id]

            # For non-stopwords, and their topic ids are not t
            if w_t != t and w_t >= 0 and w_t <= K:
                p_wt[w_id, t] = (1 - c_w) * p_w[w_id]

            # For stopwords, their topic ids locate outside [0, K]
            if w_t < 0 or w_t > K:
                p_wt[w_id, t] = p_w[w_id]

    return p_wt


def get_topic_doc_distribution_ptd(K, p_t, c_t, document_topic_assign_list):

    D = len(document_topic_assign_list)

    p_td = np.zeros((K, D))

    for d_id, d_t in enumerate(document_topic_assign_list):
        for t in range(K):
            if d_t == t:
                p_td[t, d_id] = float(c_t + (1 - c_t) * p_t[t])

            if d_t != t:
                p_td[t, d_id] = float((1 - c_t) * p_t[t])

    return p_td




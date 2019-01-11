import numpy as np
# import random
import os
import sys
from gensim import corpora
# import gzip
# from collections import Counter
import subprocess
# from common.convert_states import state_perturb_wd
from common.convert_states import nwd_to_texts, state_nwjd
from measures.overlap import state_dwz_nmi, get_dict_output_token_labeling
from measures.classify import predict_topic_p_td_unsup
from measures.coherence import topic_cherence_C



def make_path_tmp_tm():
    marker = True
    while marker is True:
        ind_tmp = np.random.randint(0, 32000)
        path_tmp = os.path.abspath(os.path.join(os.pardir, 'tmp'))
        path_tmp_name = os.path.join(path_tmp, 'tmp_tm_%s' % (str(ind_tmp)))
        if os.path.isdir(path_tmp_name):
            pass
        else:
            os.system('cd %s;mkdir tmp_tm_%s' % (str(path_tmp), str(ind_tmp)))
            marker = False
    return path_tmp_name + '/'


def texts_corpus_tm(texts, train_dir):
    fname_data = 'data_tmp_tm'
    f = open(train_dir + fname_data, 'w')
    for text in texts:
        for w in text:
            f.write(w + ' ')
        f.write('\n')
    f.close()
    return train_dir + fname_data


def tm_inference(path_tm, texts):
    train_dir = make_path_tmp_tm()
    train_fname = texts_corpus_tm(texts, train_dir)
    dir_cwd = os.getcwd()

    os.chdir(path_tm)
    cmd_tm = './bin/topicmap -f %s -o %stest_result' % (train_fname, train_dir)
    # os.system(cmd_tm)
    p = subprocess.Popen(cmd_tm, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    # # number of topics
    f = open(train_dir + 'test_result/lda_class_words.txt', 'r')
    x = f.readlines()
    f.close()
    K_tm = len(x)

    # # number of types
    f = open(train_dir + 'test_result/word_wn_count.txt', 'r')
    x = f.readlines()
    f.close()
    V_ = len(x)

    f = open(train_dir + 'test_result/lda_word_assignments_final.txt')
    x = f.readlines()
    f.close()
    D_ = len(x)

    # # word-id - mapping
    f = open(train_dir + 'test_result/word_wn_count.txt')
    x = f.readlines()
    f.close()
    list_words = [int(h.split()[0]) for h in x]
    list_words_id = [int(h.split()[1]) for h in x]
    dict_id_word = dict(zip(list_words_id, list_words))

    n_wj_ = np.zeros((V_, K_tm)).astype('int')
    n_jd_ = np.zeros((K_tm, D_)).astype('int')

    state_dwz_tm = []
    for d_, x_tmp in enumerate(x):
        h = [int(h_) for h_ in x_tmp.split()]
        n_h = len(h)
        s_h = int(n_h / 4)
        for i_ in range(s_h):
            # w_ = h[i_*4]
            w_ = dict_id_word
            n_ = h[(i_) * 4 + 2]
            z_ = h[(i_) * 4 + 3]
            state_dwz_tm += [(d_, w_, z_)] * n_
            n_wj_[w_, z_] += 1
            n_jd_[z_, d_] += 1

    # state_dwz_tm_s,state_dwz_tm_r = state_perturb_wd(state_dwz_tm)

    os.system('rm -rf %s' % (train_dir))
    return state_dwz_tm, n_wj_, n_jd_, K_tm


def tm_inference_p_dt(path_tm, texts):
    train_dir = make_path_tmp_tm()
    train_fname = texts_corpus_tm(texts, train_dir)
    dir_cwd = os.getcwd()

    os.chdir(path_tm)
    cmd_tm = './bin/topicmap -f %s -o %stest_result' % (train_fname, train_dir)
    # os.system(cmd_tm)
    p = subprocess.Popen(cmd_tm, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    f = open(train_dir + 'test_result/lda_gammas_final.txt')
    x = f.readlines()
    f.close()
    D_ = len(x)
    K_tm = len(x[0].split())

    theta_dj_tm = np.zeros((D_, K_tm))
    for i_d, d in enumerate(x):
        n_j_tmp = np.array([float(h_) for h_ in d.split()])
        theta_dj_tm[i_d, :] = n_j_tmp / float(np.sum(n_j_tmp))
    os.system('rm -rf %s' % (train_dir))

    # os.system('rm -rf %s'%(train_dir))
    return theta_dj_tm


def pp_inference_tm(state_dwz_, dict_params_, dict_args_):
    try:
        path_tm_ = dict_args_['path_tm']
    except KeyError:
        print('specify path to tm. default might not work.')
    # # parameters
    D_ = dict_params_['D']
    V_ = dict_params_['V']
    K_ = dict_params_['K']

    # # convert state into corpus
    n_wd_, n_wj_, n_jd_ = state_nwjd(state_dwz_, D_, V_, K_)
    texts_ = nwd_to_texts(n_wd_)

    state_dwz_infer_, n_wj_infer_, n_jd_infer_, K_infer_ = tm_inference(path_tm_, texts_)
    # # get the nmi and the rest
    nmi = state_dwz_nmi(state_dwz_, state_dwz_infer_, K_, K_infer_)

    # #
    p_dt_infer = np.transpose(n_jd_infer_ / float(np.sum(n_jd_infer_)))

    list_t_d_infer = predict_topic_p_td_unsup(p_dt_infer)

    return nmi, K_infer_, p_dt_infer, list_t_d_infer


def tm_inference_wrapper(dict_input):
    '''
    Wrapper for tm_inference

    Input:
        dict_input = {
            ## choose topic model
            'topic_model': 'tm'

            ## provide corpus and number of topics if need
            , 'texts':texts

            ## optional, only works for synthetic corpus with token labeling
            , 'state_dwz_true': state_dwz
            , 'k_true': K

            ## optional
            , 'path_tm': os.path.abspath(os.path.join(os.pardir,'src/external/topicmapping'))
        }

    Output:
        dict_output = {
            'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': inferred number of topics
        }
    '''
    # ############################
    # # Get input parameters
    texts = dict_input['texts']
    flag_coherence = dict_input.get('flag_coherence', 0)

    # # optional, only works for synthetic corpus with token labeling
    state_dwz_true = dict_input.get('state_dwz_true', None)
    k_true = dict_input.get('k_true', None)

    # # optional
    path_tm = dict_input.get('path_tm', os.path.abspath(os.path.join(os.pardir, 'src/external/topicmapping')))

    # # Call the true function:
    dict_output = tm_inference_terminal(texts, state_dwz_true=state_dwz_true, k_true=k_true, flag_coherence=flag_coherence, path_tm=path_tm)

    return dict_output


def tm_inference_terminal(texts, state_dwz_true=None, k_true=None, flag_coherence=0, path_tm=os.path.abspath(os.path.join(os.pardir, 'src/external/topicmapping'))):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:

        ## provide corpus and number of topics if need
        , 'texts':texts

        ## optional, only works for synthetic corpus with token labeling
        , 'state_dwz_true': state_dwz
        , 'k_true': K

        ## optional
        , 'path_tm': os.path.abspath(os.path.join(os.pardir,'src/external/topicmapping'))


    Output:
        dict_output = {
            'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': inferred number of topics
        }
    '''

    #############################
    # # Generate a empty dic for output
    dict_output = {}

    #############################
    # # inference for p_dt

    train_dir = make_path_tmp_tm()
    train_fname = texts_corpus_tm(texts, train_dir)
    dir_cwd = os.getcwd()

    os.chdir(path_tm)
    cmd_tm = './bin/topicmap -f %s -o %stest_result' % (train_fname, train_dir)
    # os.system(cmd_tm)
    p = subprocess.Popen(cmd_tm, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    # ############################
    # # get p_td_tm:

    p_td_tm = tm_inference_get_p_td_tm(train_dir)
    dict_output['p_td_infer'] = p_td_tm

    # ############################
    # # get p_wt_tm:
    p_wt_tm = tm_inference_get_p_wt_tm(train_dir)
    dict_output['p_wt_infer'] = p_wt_tm

    # ############################
    # # get the number of topics:
    f = open(train_dir + 'test_result/lda_class_words.txt', 'r')
    x = f.readlines()
    f.close()
    k_tm = len(x)
    dict_output['k_infer'] = k_tm

    if flag_coherence == 1:
        state_dwz_tm = tm_inference_get_state_dwz_tm(train_dir)
        dict_gs = corpora.Dictionary(texts)
        all_terms = list(dict_gs.iterkeys())
        V = len(all_terms)
        D = len(texts)
        n_wd_, n_wj_, n_jd_ = state_nwjd(state_dwz_tm, D, V, k_tm)
        dict_output['coherence'] = topic_cherence_C(n_wd_, n_wj_)

    # ############################

    # # Get the nmi for token_labeling
    state_dwz_tm = tm_inference_get_state_dwz_tm(train_dir)
    dict_output['state_dwz_infer'] = state_dwz_tm

    if state_dwz_true is not None:

        dict_output_token_labeling = get_dict_output_token_labeling(state_dwz_true, state_dwz_tm, k_true, k_tm)

        dict_output.update(dict_output_token_labeling)

    os.system('rm -rf %s' % (train_dir))

    return dict_output


def tm_inference_get_p_td_tm(train_dir):

    f = open(train_dir + 'test_result/lda_gammas_final.txt')
    x = f.readlines()
    f.close()
    D_ = len(x)
    K_tm = len(x[0].split())

    p_td_tm = np.zeros((D_, K_tm))
    for i_d, d in enumerate(x):
        n_j_tmp = np.array([float(h_) for h_ in d.split()])
        p_td_tm[i_d, :] = n_j_tmp / float(np.sum(n_j_tmp))

    return p_td_tm


def tm_inference_get_p_wt_tm(train_dir):

    # # word-id - mapping
    f = open(train_dir + 'test_result/word_wn_count.txt')
    x = f.readlines()
    f.close()
    list_words = [int(h.split()[0]) for h in x]
    list_words_id = [int(h.split()[1]) for h in x]
    dict_id_word = dict(zip(list_words_id, list_words))
    n_w = max(list_words) + 1

    f = open(train_dir + 'test_result/lda_betas_sparse_final.txt')
    x = f.readlines()
    f.close()

    K_tm = len(x)
    K_tm

    p_wt_tm = np.zeros([n_w, K_tm])

    for tmp_K, tmp_x in enumerate(x):

        tmp_x = x[tmp_K]
        tmp_x_list = tmp_x.split()

        word_pair_num = (len(tmp_x_list) - 1) / 2
        for word_pair_num in range(int(word_pair_num)):
            tmp_word = int(tmp_x_list[2 * word_pair_num + 1])
            tmp_p_word = float(tmp_x_list[2 * word_pair_num + 2])
            p_wt_tm[dict_id_word[tmp_word], tmp_K] = tmp_p_word

    p_wt_tm = p_wt_tm / p_wt_tm.sum(axis=0)

    return p_wt_tm


def tm_inference_get_state_dwz_tm(train_dir):

    # # word-id - mapping
    f = open(train_dir + 'test_result/word_wn_count.txt')
    x = f.readlines()
    f.close()
    list_words = [int(h.split()[0]) for h in x]
    list_words_id = [int(h.split()[1]) for h in x]
    dict_id_word = dict(zip(list_words_id, list_words))

    f = open(train_dir + 'test_result/lda_word_assignments_final.txt')
    x = f.readlines()
    f.close()
    # # individual labels
    state_dwz_tm = []
    for d_, x_tmp in enumerate(x):
        h = [int(h_) for h_ in x_tmp.split()]
        n_h = len(h)
        s_h = int(n_h / 4)
        for i_ in range(s_h):
            # w_ = h[i_*4]
            w_ = dict_id_word[h[i_ * 4]]
            n_ = h[(i_) * 4 + 2]
            z_ = h[(i_) * 4 + 3]
            state_dwz_tm += [(d_, w_, z_)] * n_

    return state_dwz_tm

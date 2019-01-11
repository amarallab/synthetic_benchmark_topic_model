import numpy as np
# import random
import os
import sys
# import gzip
from gensim import corpora
from collections import Counter
import subprocess
src_dir = os.path.abspath(os.path.join(os.pardir, 'src'))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# from common.convert_states import state_perturb_wd
from common.convert_states import nwd_to_texts, state_nwjd
from measures.overlap import state_dwz_nmi, get_dict_output_token_labeling
from measures.classify import predict_topic_p_td_unsup
from measures.coherence import topic_cherence_C


def make_path_tmp_hdp():
    marker = True
    while marker is True:
        ind_tmp = np.random.randint(0, 32000)
        path_tmp = os.path.abspath(os.path.join(os.pardir, 'tmp'))
        path_tmp_name = os.path.join(path_tmp, 'tmp_hdp_%s' % (str(ind_tmp)))
        if os.path.isdir(path_tmp_name):
            pass
        else:
            os.system('cd %s;mkdir tmp_hdp_%s' % (str(path_tmp), str(ind_tmp)))
            marker = False
    return path_tmp_name + '/'


def texts_corpus_hdp(texts, train_dir):
    fname_data = 'data_tmp'
    f = open(train_dir + fname_data, 'w')
    for text in texts:
        c_text = Counter(text)
        V_d = len(c_text)
        list_w_text = list(c_text.keys())
        list_w_text.sort()
        f.write(str(V_d))
        for w_ in list_w_text:
            f.write(' ' + str(w_) + ':' + str(c_text[w_]))
        f.write('\n')
    f.close()
    return train_dir + fname_data


def hdp_inference(path_hdp, texts):

    train_dir = make_path_tmp_hdp()
    train_fname = texts_corpus_hdp(texts, train_dir)
    dir_cwd = os.getcwd()
    os.chdir(path_hdp)

    cmd_hdp = './hdp --train_data %s --directory %s' % (train_fname, train_dir)

    p = subprocess.Popen(cmd_hdp, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    # # word-topic counts
    f = open(train_dir + 'final.topics', 'r')
    x = f.readlines()
    f.close()
    K_hdp = len(x)
    V_ = len(x[0].split())
    n_wj_hdp = np.zeros((V_, K_hdp)).astype('int')
    for i_j, j in enumerate(x):
        n_wj_tmp = np.array([int(h_) for h_ in j.split()])
        n_wj_hdp[:, i_j] = n_wj_tmp

    # #  doc-topic counts
    f = open(train_dir + 'final.doc.states', 'r')
    x = f.readlines()
    f.close()
    D_ = len(x)
    n_jd_hdp = np.zeros((K_hdp, D_)).astype('int')
    for i_d, d in enumerate(x):
        n_jd_tmp = np.array([int(h_) for h_ in d.split()])
        n_jd_hdp[:, i_d] = n_jd_tmp

    # # individual labels
    f = open(train_dir + 'final.word-assignments', 'r')
    # header = f.readline()
    x = f.readlines()
    f.close()
    state_dwz_hdp = [tuple([int(h_) for h_ in h.split()]) for h in x]
    # state_dwz_hdp_s,state_dwz_hdp_r = state_perturb_wd(state_dwz_hdp)
    os.system('rm -rf %s' % (train_dir))
    return state_dwz_hdp, n_wj_hdp, n_jd_hdp, K_hdp


def hdp_inference_p_dt(path_hdp, texts):

    train_dir = make_path_tmp_hdp()
    train_fname = texts_corpus_hdp(texts, train_dir)
    dir_cwd = os.getcwd()
    os.chdir(path_hdp)

    cmd_hdp = './hdp --train_data %s --directory %s' % (train_fname, train_dir)

    p = subprocess.Popen(cmd_hdp, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    # # doc-topic counts
    f = open(train_dir + 'final.doc.states', 'r')
    x = f.readlines()
    f.close()
    D_ = len(x)
    K_hdp = len(x[0].split())
    p_td_hdp = np.zeros((D_, K_hdp))
    for i_d, d in enumerate(x):
        n_j_tmp = np.array([int(h_) for h_ in d.split()])
        p_td_hdp[i_d, :] = n_j_tmp / float(np.sum(n_j_tmp))
    os.system('rm -rf %s' % (train_dir))

    # ### TODO: Any smoothing with alpha or something to obtain theta???

    return p_td_hdp


def pp_inference_hdp(state_dwz_, dict_params_, dict_args_):
    try:
        path_hdp_ = dict_args_['path_hdp']
    except KeyError:
        print('specify path to hdp. default might not work.')
    # # parameters
    D_ = dict_params_['D']
    V_ = dict_params_['V']
    K_ = dict_params_['K']

    # # convert state into corpus
    n_wd_, n_wj_, n_jd_ = state_nwjd(state_dwz_, D_, V_, K_)
    texts_ = nwd_to_texts(n_wd_)

    state_dwz_infer_, n_wj_infer_, n_jd_infer_, K_infer_ = hdp_inference(path_hdp_, texts_)
    # # get the nmi and the rest
    nmi = state_dwz_nmi(state_dwz_, state_dwz_infer_, K_, K_infer_)

    # #
    p_dt_infer = np.transpose(n_jd_infer_ / float(np.sum(n_jd_infer_)))

    list_t_d_infer = predict_topic_p_td_unsup(p_dt_infer)

    return nmi, K_infer_, p_dt_infer, list_t_d_infer


def hdp_inference_wrapper(dict_input):
    '''
    Wrapper for hdp_inference

    Input:
        dict_input = {
            ## choose topic model
            'topic_model': 'hdp'

            ## provide corpus and number of topics if need
            , 'texts':texts

            ## optional, only works for synthetic corpus with token labeling
            , 'state_dwz_true': state_dwz
            , 'k_true': K

            ## optional
            , 'path_hdp': os.path.abspath(os.path.join(os.pardir,'src/external/hdp-bleilab/hdp-faster'))
        }

    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': number of topics inferred by topic model
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
    path_hdp = dict_input.get('path_hdp', os.path.abspath(os.path.join(os.pardir, 'src/external/hdp-bleilab/hdp-faster')))

    # # Call the true function:
    dict_output = hdp_inference_terminal(texts, state_dwz_true=state_dwz_true, k_true=k_true, flag_coherence=flag_coherence, path_hdp=path_hdp)

    return dict_output


def hdp_inference_terminal(texts, state_dwz_true=None, k_true=None, flag_coherence=0, path_hdp=os.path.abspath(os.path.join(os.pardir, 'src/external/hdp-bleilab/hdp-faster'))):
    '''
    Do the inference for p_dt and  state_dwz_ (optional)

    Input:

        ## provide corpus and number of topics if need
        , 'texts':texts

        ## optional, only works for synthetic corpus with token labeling
        , 'state_dwz_true': state_dwz
        , 'k_true': K

        ## optional
        , 'path_hdp': os.path.abspath(os.path.join(os.pardir,'src/external/hdp-bleilab/hdp-faster'))


    Output:
        dict_output = {
              'p_td_infer': p_td ## p_td inferred by topic modle
            , 'token_labeling_nmi': nmi ## optional, results for token labeling, only for synthetic data
            , 'k_infer': number of topics inferred by topic model
        }
    '''

    #############################
    # # Generate a empty dic for output
    dict_output = {}

    # ############################
    # # inference for p_dt

    train_dir = make_path_tmp_hdp()
    train_fname = texts_corpus_hdp(texts, train_dir)
    dir_cwd = os.getcwd()
    os.chdir(path_hdp)

    cmd_hdp = './hdp --train_data %s --directory %s' % (train_fname, train_dir)

    p = subprocess.Popen(cmd_hdp, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()
    os.chdir(dir_cwd)

    # # doc-topic counts
    f = open(train_dir + 'final.doc.states', 'r')
    x = f.readlines()
    f.close()
    D_ = len(x)
    K_hdp = len(x[0].split())
    p_td_hdp = np.zeros((D_, K_hdp))
    for i_d, d in enumerate(x):
        n_j_tmp = np.array([int(h_) for h_ in d.split()])
        p_td_hdp[i_d, :] = n_j_tmp / float(np.sum(n_j_tmp))
    # os.system('rm -rf %s'%( train_dir))

    dict_output['p_td_infer'] = p_td_hdp

    # ############################
    # # get the number of topics:
    f = open(train_dir + 'final.topics', 'r')
    x = f.readlines()
    f.close()
    k_hdp = len(x)
    dict_output['k_infer'] = k_hdp

    # ############################
    # # individual labels
    f = open(train_dir + 'final.word-assignments', 'r')
    header = f.readline()
    x = f.readlines()
    f.close()
    state_dwz_hdp = [tuple([int(h_) for h_ in h.split()]) for h in x]
    dict_output['state_dwz_infer'] = state_dwz_hdp

    if flag_coherence == 1:
        dict_gs = corpora.Dictionary(texts)
        all_terms = list(dict_gs.iterkeys())
        V = len(all_terms)
        D = len(texts)
        n_wd_, n_wj_, n_jd_ = state_nwjd(state_dwz_hdp, D, V, k_hdp)
        dict_output['coherence'] = topic_cherence_C(n_wd_, n_wj_)

    # ##############
    # # infer p_wt

    all_word_list = [i[1] for i in state_dwz_hdp]
    n_w = max(all_word_list) + 1

    num_k = k_hdp
    p_wt_infer = np.zeros([n_w, num_k])
    for i in state_dwz_hdp:
        tmp_w = i[1]
        tmp_t = i[2]
        p_wt_infer[tmp_w, tmp_t] += 1
    p_wt_infer = p_wt_infer / p_wt_infer.sum(axis=0)
    dict_output['p_wt_infer'] = p_wt_infer

    # # Get the nmi for token_labeling
    if state_dwz_true is not None:
        dict_output_token_labeling = get_dict_output_token_labeling(state_dwz_true, state_dwz_hdp, k_true, k_hdp)
        dict_output.update(dict_output_token_labeling)

    os.system('rm -rf %s' % (train_dir))

    return dict_output

import numpy as np
from collections import Counter
from common.convert_states import *
from scipy.optimize import linear_sum_assignment
from measures.classify import *
from random import shuffle


def obtain_nmi_acc_sup(topic_list, p_td_ldavb_array, n_folds=10, n_neighbors=10, weights='uniform'):
    '''
    Obtain nmi and acc with supervised methods
    Input:
    - topic_list: list, list of topics for all documents
    - p_td_ldavb_array: array, topic propotion for each documents
    - n_folds: k-fold cross validation
    - n_neighbors: k-nearest neighbourhood algorithm
    Output:
    - nmi: normalized mutual information
    - acc: accuracy
    '''
    topic_label_real_array = np.array(topic_list)
    D = len(topic_list)

    # Supervised classification algorithm
    kf = KFold(D, n_folds=n_folds, shuffle=True)  # random_state = 5 )
    # Generate the K-nearest model
    clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    # Get the training and text index
    for train, test in kf:
        break

    train_in = p_td_ldavb_array[train]
    train_out = topic_label_real_array[train]
    # Train the K-nearest model
    clf.fit(train_in, train_out)

    # Get the input data for test, predict the output based on the trained K-nearest model
    test_in = p_td_ldavb_array[test]
    test_out = clf.predict(test_in)

    # Get the real value for the test output
    test_real = topic_label_real_array[test]

    # Calculate the nmi and acc
    nmi = calc_class_doc_nmi(test_real, test_out)
    acc = calc_class_doc_acc_unsup(test_real, test_out)

    return nmi, acc


def obtain_nmi_acc_unsup(topic_list, p_td_ldavb_array, removed_topic_list=None):
    '''
    Obtain nmi and acc with unsupervised methods
    Input:
    - topic_list: list, list of topics for all documents
    - p_td_ldavb_array: array, topic propotion for each documents
    Output:
    - nmi: normalized mutual information
    - acc: accuracy

    '''
    list_t_d_pred = predict_topic_p_td_unsup(p_td_ldavb_array)

    if removed_topic_list is not None and len(removed_topic_list) > 0:
        num_doc_removed = len(removed_topic_list)
        k_num = max(topic_list)
        random_topic_removed = list(np.random.choice(k_num, num_doc_removed))

        topic_list = list(topic_list) + list(removed_topic_list)
        list_t_d_pred = list(list_t_d_pred) + random_topic_removed

    nmi = calc_class_doc_nmi(topic_list, list_t_d_pred)
    acc = calc_class_doc_acc_unsup(topic_list, list_t_d_pred)

    return nmi, acc


def state_dwz_nmi(state_dwz1_, state_dwz2_, K1_, K2_, normalized=True):
    '''
    Calculate the normalized mutual information (NMI) between two labeled dwz-states, i.e. how much the two labeled states overlap.
    In: - state_dwz1_,
        - state_dwz2_,
        - K1_, # of topics in state_dwz1_, int
        - K2_, # of topics in state_dwz2_, int
    Out:
        - NMI, float
    '''
    # VI_ = 0.0
    N_ = len(state_dwz1_)
    n_tt_ = np.zeros((K1_, K2_))

    state_dwz1_s, state_dwz1_ = state_perturb_wd(state_dwz1_)  # sort and shuffle labels across same words and docs
    state_dwz2_s, state_dwz2_ = state_perturb_wd(state_dwz2_)  # sort and shuffle labels across same words and docs

    list_z1_z2_ = [(state_dwz1_[i_][2], state_dwz2_[i_][2]) for i_ in range(N_)]
    c_z1_z2_ = Counter(list_z1_z2_)
    for z1_z2_, n_z1_z2_ in c_z1_z2_.items():
        n_tt_[z1_z2_[0], z1_z2_[1]] += n_z1_z2_
    p_tt_ = n_tt_ / float(N_)
    p_t1_ = np.sum(p_tt_, axis=1)
    p_t2_ = np.sum(p_tt_, axis=0)
    H1_ = sum([-p_ * np.log(p_) for p_ in p_t1_ if p_ > 0.0])
    H2_ = sum([-p_ * np.log(p_) for p_ in p_t2_ if p_ > 0.0])
    MI_ = 0.0
    for i_ in range(K1_):
        for j_ in range(K2_):
            p1_ = p_t1_[i_]
            p2_ = p_t2_[j_]
            p12_ = p_tt_[i_, j_]
            if p12_ > 0.0:
                MI_ += p12_ * np.log(p12_ / (p1_ * p2_))
    if normalized is True:
        NMI_ = 2.0 * MI_ / (H1_ + H2_)
    else:
        NMI_ = 1.0 * MI_
        # if we want to return the unnormalized mutual information
    return NMI_


def state_dwz_normal_nmi(state_dwz_true, state_dwz_infer, k_true, k_infer):
    '''Calculate the normalized 'normalized mutual information' (NMI) between two labeled dwz-states,
        i.e. how much the two labeled states overlap compared with the randon dwz-states
    In: - state_dwz_true,
        - state_dwz_infer,
        - k_true, # of topics in state_dwz_true, int
        - k_infer, # of topics in state_dwz_infer (inferred or input), int
    Out:
        - normal_nmi, float
    '''

    # Get the true_token_topic_list
    true_token_topic_list = []

    for i in range(len(state_dwz_true)):
        true_token_topic_list.append(state_dwz_true[i][2])

    # shuffle true_token_topic_list
    shuffle(true_token_topic_list)

    # Get state_dwz_true_shuffle
    state_dwz_true_shuffle = []
    for i in range(len(state_dwz_true)):
        state_dwz_true_shuffle.append((state_dwz_true[i][0], state_dwz_true[i][1], true_token_topic_list[i]))

    # Get perfect_nmi, rand_nmi, model_nmi, normal_nmi
    perfect_nmi = state_dwz_nmi(state_dwz_true, state_dwz_true, k_true, k_true)
    rand_nmi = state_dwz_nmi(state_dwz_true, state_dwz_true_shuffle, k_true, k_true)
    model_nmi = state_dwz_nmi(state_dwz_true, state_dwz_infer, k_true, k_infer)
    normal_nmi = (model_nmi - rand_nmi) / (perfect_nmi - rand_nmi)

    # non-normalized mi
    perfect_mi = state_dwz_nmi(state_dwz_true, state_dwz_true, k_true, k_true, normalized=False)
    rand_mi = state_dwz_nmi(state_dwz_true, state_dwz_true_shuffle, k_true, k_true, normalized=False)
    model_mi = state_dwz_nmi(state_dwz_true, state_dwz_infer, k_true, k_infer, normalized=False)
    normal_mi = (model_mi - rand_mi) / (perfect_mi - rand_mi)

    dict_output = {}
    dict_output['perfect_nmi'] = perfect_nmi
    dict_output['rand_nmi'] = rand_nmi
    dict_output['model_nmi'] = model_nmi
    dict_output['normal_nmi'] = normal_nmi

    dict_output['perfect_mi'] = perfect_mi
    dict_output['rand_mi'] = rand_mi
    dict_output['model_mi'] = model_mi
    dict_output['normal_mi'] = normal_mi

    return dict_output


def get_dict_output_token_labeling(state_dwz_true, state_dwz_infer, k_true, k_infer):
    nmi_dict = state_dwz_normal_nmi(state_dwz_true, state_dwz_infer, k_true, k_infer)
    dict_output_nmi = {}
    dict_output_nmi['token_labeling_perfect_nmi'] = nmi_dict['perfect_nmi']
    dict_output_nmi['token_labeling_rand_nmi'] = nmi_dict['rand_nmi']
    dict_output_nmi['token_labeling_model_nmi'] = nmi_dict['model_nmi']
    dict_output_nmi['token_labeling_normal_nmi'] = nmi_dict['normal_nmi']

    # dict_output_nmi['token_labeling_perfect_mi'] = nmi_dict['perfect_mi']
    # dict_output_nmi['token_labeling_rand_mi'] = nmi_dict['rand_mi']
    # dict_output_nmi['token_labeling_model_mi'] = nmi_dict['model_mi']
    # dict_output_nmi['token_labeling_normal_mi'] = nmi_dict['normal_mi']

    # # # by martin: comparing n(w,d,t) via jensen-shannon-divergence
    # jsd = state_dwz_jsd(state_dwz_true, state_dwz_infer, k_true, k_infer)
    # dict_output_nmi['jsd'] = jsd

    # state_dwz_true_rand = randomize_tokenlabels_state_dwz(state_dwz_true)
    # state_dwz_infer_rand = randomize_tokenlabels_state_dwz(state_dwz_infer)
    # jsd_rand = state_dwz_jsd(state_dwz_true, state_dwz_true_rand, k_true, k_true)
    # dict_output_nmi['jsd_rand'] = jsd_rand

    # # # true-infer
    # jsd_tmp = state_dwz_jsd(state_dwz_true, state_dwz_infer, k_true, k_infer)
    # dict_output_nmi['jsd_t_i'] = jsd_tmp
    # # # true_r-infer_r
    # jsd_tmp = state_dwz_jsd(state_dwz_true_rand, state_dwz_infer_rand, k_true, k_infer)
    # dict_output_nmi['jsd_tr_ir'] = jsd_tmp
    # # # true|rand-infer
    # jsd_tmp = state_dwz_jsd(state_dwz_true_rand, state_dwz_infer, k_true, k_infer)
    # dict_output_nmi['jsd_tr_i'] = jsd_tmp
    # # # true-infer|rand
    # jsd_tmp = state_dwz_jsd(state_dwz_true, state_dwz_infer_rand, k_true, k_infer)
    # dict_output_nmi['jsd_t_ir'] = jsd_tmp
    # # # true-true|rand
    # jsd_tmp = state_dwz_jsd(state_dwz_true, state_dwz_true_rand, k_true, k_true)
    # dict_output_nmi['jsd_t_tr'] = jsd_tmp
    # # # infer-infer|rand
    # jsd_tmp = state_dwz_jsd(state_dwz_infer, state_dwz_infer_rand, k_infer, k_infer)
    # dict_output_nmi['jsd_i_ir'] = jsd_tmp

    return dict_output_nmi


def get_nmi_for_null(state_dwz_true, k_true):

    # Get state_dwz_null_rand
    state_dwz_null_rand = []
    null_rand_token_topic_list = list(np.random.choice(k_true, len(state_dwz_true)))
    for i in range(len(state_dwz_true)):
        state_dwz_null_rand.append((state_dwz_true[i][0], state_dwz_true[i][1], null_rand_token_topic_list[i]))

    # Get state_dwz_null_rand
    state_dwz_null_one = []
    null_one_token_topic_list = list(np.random.choice(1, len(state_dwz_true)))
    for i in range(len(state_dwz_true)):
        state_dwz_null_one.append((state_dwz_true[i][0], state_dwz_true[i][1], null_one_token_topic_list[i]))

    # null_one, null_rand
    null_one_nmi = state_dwz_nmi(state_dwz_true, state_dwz_null_one, k_true, k_true)
    null_rand_nmi = state_dwz_nmi(state_dwz_true, state_dwz_null_rand, k_true, k_true)

    dict_output = {}
    dict_output['null_one_nmi'] = null_one_nmi
    dict_output['null_rand_nmi'] = null_rand_nmi

    return dict_output


def calc_class_doc_nmi(list_t_true, list_t_pred):
    '''
    Claculate norm mutual information between true and predicted topics of documents
    IN: two list of len=D where each entry is the true and pred topic of the respective document
    OUT: nmi (float)
    '''
    K1 = max(list_t_true) + 1  # number of topics in true
    K2 = max(list_t_pred) + 1  # number of topics in pred
    N = len(list_t_true)
    n_tt = np.zeros((K1, K2))
    list_z1_z2 = [(list_t_true[i], list_t_pred[i]) for i in range(N)]
    c_z1_z2 = Counter(list_z1_z2)
    for z1_z2_, n_z1_z2_ in c_z1_z2.items():
        n_tt[z1_z2_[0], z1_z2_[1]] += n_z1_z2_
    p_tt = n_tt / float(N)
    p_t1 = np.sum(p_tt, axis=1)
    p_t2 = np.sum(p_tt, axis=0)
    H1 = sum([-p_ * np.log(p_) for p_ in p_t1 if p_ > 0.0])
    H2 = sum([-p_ * np.log(p_) for p_ in p_t2 if p_ > 0.0])
    MI = 0.0
    for i_ in range(K1):
        for j_ in range(K2):
            p1_ = p_t1[i_]
            p2_ = p_t2[j_]
            p12_ = p_tt[i_, j_]
            if p12_ > 0.0:
                MI += p12_ * np.log(p12_ / (p1_ * p2_))
    NMI = 2.0 * MI / (H1 + H2)
    return NMI


def calc_class_doc_acc_unsup(list_t_true, list_t_pred):
    '''Claculate accuracy between true and predicted topics of documents
    IN: two list of len=D where each entry is the true and pred topic of the respective document
    OUT: acc = fraction of correctly predicted documents(float)

    Note that in the unsupervised prediction, the number of the labels is arbitrary; hence
    we have to match true and inferred topics.
    We use the Kuhn-Menkres algorithm; which finds the best matching of true and inferred topics
    see here: http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    '''
    K1 = max(list_t_true) + 1  # number of topics in true
    K2 = max(list_t_pred) + 1  # number of topics in pred
    N = len(list_t_true)
    n_tt = np.zeros((K1, K2))
    list_z1_z2 = [(list_t_true[i], list_t_pred[i]) for i in range(N)]
    c_z1_z2 = Counter(list_z1_z2)
    for z1_z2_, n_z1_z2_ in c_z1_z2.items():
        n_tt[z1_z2_[0], z1_z2_[1]] += n_z1_z2_

    # the kuhn-menkres finds the minimum: -1*n_tt
    row_ind, col_ind = linear_sum_assignment(-1.0 * n_tt)
    acc = n_tt[row_ind, col_ind].sum() / float(len(list_t_true))
    return acc


def calc_class_doc_acc_sup(list_t_true, list_t_pred):
    '''Claculate accuracy between true and predicted topics of documents
    IN: two list of len=D where each entry is the true and pred topic of the respective document
    OUT: acc = fraction of correctly predicted documents(float)

    These are the predicted albel from supervised classification ,
    so we only count the fraction of documents that have the same label in true and predicted.
    '''
    K1 = max(list_t_true) + 1  # number of topics in true
    K2 = max(list_t_pred) + 1  # number of topics in pred
    N = len(list_t_true)
    n_tt = np.zeros((K1, K2))

    list_z1_z2 = [(list_t_true[i], list_t_pred[i]) for i in range(N)]
    c_z1_z2 = Counter(list_z1_z2)
    for z1_z2_, n_z1_z2_ in c_z1_z2.items():
        n_tt[z1_z2_[0], z1_z2_[1]] += n_z1_z2_

    acc = 0
    for i in range(min([K1, K2])):
        acc += n_tt[i, i]
    acc /= float(N)

    return acc


def state_dwz_jsd(state_dwz1, state_dwz2, K1, K2):
    c1 = Counter(state_dwz1)
    c2 = Counter(state_dwz2)
    N = len(state_dwz1)

    list_wd = sorted(list(set([(h[0], h[1]) for h in c1.keys()])))
    S = len(list_wd)
    dict_wd_s = dict(zip(list_wd, np.arange(S)))

    T = np.max([K1, K2])
    # Tmin = np.min([K1, K2])
    # Tmin_ind = np.argmin([K1, K2])
    arr_st1 = np.zeros((S, T))
    arr_st2 = np.zeros((S, T))

    for (w, d, z), n_wdz in c1.items():
        arr_st1[dict_wd_s[(w, d)], z] = n_wdz / N
    for (w, d, z), n_wdz in c2.items():
        arr_st2[dict_wd_s[(w, d)], z] = n_wdz / N

    D_tt = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            D_tt[i, j] = dist_p1_p2(arr_st1[:, i], arr_st2[:, j])

    JSD = D_tt[linear_sum_assignment(D_tt)].sum()
    return JSD


def randomize_tokenlabels_state_dwz(state_dwz_true):
    # Get the true_token_topic_list
    true_token_topic_list = []

    for i in range(len(state_dwz_true)):
        true_token_topic_list.append(state_dwz_true[i][2])

    # shuffle true_token_topic_list
    shuffle(true_token_topic_list)

    # Get state_dwz_true_shuffle
    state_dwz_true_shuffle = []
    for i in range(len(state_dwz_true)):
        state_dwz_true_shuffle.append((state_dwz_true[i][0], state_dwz_true[i][1], true_token_topic_list[i]))
    return state_dwz_true_shuffle


def dist_p1_p2(P1, P2):
    P12 = 0.5 * (P1 + P2)
    H12 = -np.nansum(P12 * np.log(P12))
    H1 = -np.nansum(P1 * np.log(P1))
    H2 = -np.nansum(P2 * np.log(P2))
    JSD = H12 - 0.5 * H1 - 0.5 * H2
    return JSD / np.log(2)

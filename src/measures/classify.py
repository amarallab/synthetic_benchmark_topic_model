import numpy as np
from collections import Counter
from scipy import optimize
# import gensim as gs
import random
from sklearn.cross_validation import KFold
from sklearn import neighbors
import json
import matplotlib.pyplot as plt
# from models.lda import *


######################
# This function is not used any more
######################

def obtain_p_t_d_ldamt(texts,V,K,N_iter=1000,dN_opt=0):
    '''Infer the individual token-labels for the inferred topic-doc and word-topic matrices from mallet-LDA.
    In: 
        - texts, a list of len=# of docs, i.e. texts = [text1, text2, ... textD] with
          texti is a list of len=textlength, i.e. texti=[w1, w2, ..]
        - V, # of word-types
        - K, # of topics
        - N_iter (default=1000), number of itertations in gibbs sampling
        - dN_opt (default=0), hyperparameter optimization every dN_opt itertations 
            (if you dont want to optimize hyperparameters put to 0)
    Out:
        - state_dwz, inferred dwz-state list, the tokens are in the same order as state_dwz
        - n_jd, inferred topic-doc count matrix, array KxD
        - n_wj, inferred word-topic count matrix, array VxK
    '''
    path_mallet = os.path.abspath(os.path.join(os.pardir,'src/external/mallet-2.0.8RC3/bin/mallet'))

    dict_gs = gs.corpora.Dictionary([[str(i)] for i in range(V)])
    corpus_gs = [dict_gs.doc2bow(text) for text in texts]

    V = len(dict_gs)
    D = len(corpus_gs)
    
    path_tmp = make_path_tmp_lda()
    model = LdaMallet(path_mallet,corpus_gs,\
                                         num_topics=K, \
                                         id2word=dict_gs,prefix=path_tmp,\
                                         iterations=N_iter,\
                                         optimize_interval = dN_opt,\
                                         workers=1)
    fdoctopics_path = model.fdoctopics()
    with open(fdoctopics_path, "r") as text_file:
        lines = text_file.readlines()
    p_t_d_ldamallet = np.zeros([D,K])
    
    for d_num in range(D):
        t_d_oneline_str = lines[d_num]
        t_d_oneline_list = t_d_oneline_str.strip('\n').split('\t')[2:]
        for t_num in range(K):
            p_t_d_ldamallet[d_num, t_num] = t_d_oneline_list[t_num]

    os.system('rm -rf %s'%(path_tmp))
    return p_t_d_ldamallet


def save_data_plot_model(path_data, path_figure, file_name, model_name, accuracy_unsuper_list,nmi_unsuper_list,accuracy_super_list, nmi_super_list, array_c_w ):
    '''
    Save the raw data to 'path_data'; Save the figure to 'path_figure'.
    In:
    - path_data: path to save data
    - path_figure: path to save figure
    - file_name: name which specifies the number of parameters
    - model_name: name of topic model
    - accuracy_unsuper_list: accuracy for unsupervised method, raw data, list of list
    - nmi_unsuper_list: nmi for unsupervised method, raw data, list of list
    - accuracy_super_list: accuracy for supervised method, raw data, list of list
    - nmi_super_list: nmi for supervised method, raw data, list of list
    - array_c_w: array c_w 
    Out:
    '''

    # Save the data
    save_all_data_dic = {}
    save_all_data_dic['accuracy_unsuper_list'] = accuracy_unsuper_list
    save_all_data_dic['nmi_unsuper_list'] = nmi_unsuper_list
    save_all_data_dic['accuracy_super_list'] = accuracy_super_list
    save_all_data_dic['nmi_super_list'] = nmi_super_list


    data_path_file = path_data + file_name + '_' + model_name + '.json'
    with open(data_path_file, 'w') as fp:
        json.dump(save_all_data_dic, fp)

    # Plot the figure

    plt.figure(figsize=(15,10))
    subplot_acc_nmi( array_c_w, accuracy_unsuper_list,  [2,2], [0,0], '$C_w$', 'Accuracy',  'accuracy_unsuper')
    subplot_acc_nmi( array_c_w, nmi_unsuper_list,       [2,2], [0,1], '$C_w$', 'nmi',       'nmi_unsuper')
    subplot_acc_nmi( array_c_w, accuracy_super_list,    [2,2], [1,0], '$C_w$', 'Accuracy',  'accuracy_super')
    subplot_acc_nmi( array_c_w, nmi_super_list,         [2,2], [1,1], '$C_w$', 'nmi',       'nmi_super')

    plt.subplots_adjust(hspace=0.3,wspace=0.3)

    figur_title = file_name + '_' + model_name 
    plt.suptitle(figur_title)

    figure_path_title = path_figure + figur_title + '.png'
    plt.savefig(figure_path_title)

    plt.close()  

    return 1

def subplot_acc_nmi( array_c_w ,raw_data_list,x_y_num_list, sub_x_y_list, xlabel_tmp, ylabel_tmp, sub_title):
    '''
    Subplot the 95CI for accuracy and 
    In:
    - array_c_w: array c_w 
    - raw_data_list: raw_data, list of list
    - x_y_num_list: [x_num, y_num], [ # of subplot in x-axis, # of subplot in y-axis]
    - sub_x_y_list: [sub_x, sub_y], position of the subplot
    - xlabel_tmp: string, xlabel for the subplot
    - ylabel_tmp: string, ylabel for the subplot
    - sub_title: string, subtitle for the subplot
    Out:
    '''
    [x_num, y_num] = x_y_num_list
    [sub_x,sub_y] = sub_x_y_list
    

    raw_avg_array = np.mean( raw_data_list,axis = 1)
    raw_low_array = np.percentile(raw_data_list, 2.5, interpolation='lower',axis = 1)
    raw_up_array = np.percentile(raw_data_list, 97.5, interpolation='higher',axis = 1)
    
    raw_low_err_array = np.array(raw_avg_array) - np.array(raw_low_array)
    raw_up_err_array  = np.array(raw_up_array)  - np.array(raw_avg_array) 

    fig = plt.subplot(x_num, y_num, sub_x * y_num +  (sub_y+1))
    fig.errorbar(array_c_w , raw_avg_array, yerr=[raw_low_err_array, raw_up_err_array])
    fig.set_title(sub_title , y = 1.05)
    fig.set_xlim(-0.05, 1.05)
    fig.set_ylim(-0.05, 1.05)
    fig.set_xlabel(xlabel_tmp)
    fig.set_ylabel(ylabel_tmp)


    return 1
    

def obtain_acc_nmi_super(p_d_t_model, list_doc_type_pp, K, D, n_folds = 10, n_neighbors = 10,weights = 'uniform' ):
    '''
    Get the accuracy and nmi based on the p_d_t (topic distribution for document) returned by specific topic model with unsuperwised algorithm
    In:
    - p_d_t_model: topic distribution for document returned by specific topic model 
    - list_doc_type_pp: planted document type
    - K: number of topics
    - D: # of documents
    - n_folds: # of fold for k-fold
    - n_neighbors: # of neighbors for k-nearest method
    - weights:the weights for k-nearest data points
    Out:
    - acc_super_model_tem: accuracy
    - nmi_super_model_tem: nmi
    '''
    
    array_doc_type_pp = np.array(list_doc_type_pp) 

    ### Supervised classification algorithm
    kf = KFold(D, n_folds = n_folds , shuffle = True) #,random_state = 5 )
    # Generate the K-nearest model
    clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)            

    accuracy_list_tmp = []
    nmi_list_tmp = []
    for train, test in kf:
        # Get the input and output for trainning 
        train_in = p_d_t_model[train]
        train_out = array_doc_type_pp[train]

        # Train the K-nearest model
        clf.fit(train_in, train_out)

        # Get the input data for test, predict the output based on the trained K-nearest model
        test_in = p_d_t_model[test]
        test_out = clf.predict(test_in)

        # Get the real value for the test output
        test_real  = array_doc_type_pp[test]

        # Get the accuracy for one specific fold and save it into list
        accuracy_tem = sum(test_real == test_out) / float(len(test_out))
        # accuracy_list_tmp.append(accuracy_tem)
         # Get the nmi for one specific fold and save it into list
        nmi_tem = calculate_nmi(test_real,test_out, K)
        # nmi_list_tmp.append(nmi_tem)       
        break
    acc_super_model_tem = accuracy_tem
    nmi_super_model_tem = nmi_tem


    return acc_super_model_tem, nmi_super_model_tem 




def obtain_acc_nmi_unsuper(p_d_t_model,list_doc_type_pp,K):
    '''
    Get the accuracy and nmi based on the p_d_t (topic distribution for document) returned by specific topic model with unsuperwised algorithm
    In:
    - p_d_t_model: topic distribution for document returned by specific topic model 
    - list_doc_type_pp: planted document type
    - K: number of topics
    Out:
    - acc_unsuper_model_tem: accuracy
    - nmi_unsuper_model_tem: nmi
    '''


    list_doc_type_model = find_doc_topic(p_d_t_model)
    acc_unsuper_model_tem = calculate_accuracy(list_doc_type_pp, list_doc_type_model, K)
    nmi_unsuper_model_tem = calculate_nmi(list_doc_type_pp,list_doc_type_model, K )
    return acc_unsuper_model_tem, nmi_unsuper_model_tem 





def obtain_p_d_t_lda(texts,V, K, dN_opt = 0):
    '''
    Get the topic distribution for each document by ldavb
    In:
    - texts: tests generated according to predetermined word distribution
    - V: number of words
    - K: number of topics
    Out:
    - p_d_t_lda: topic distribution for each document
    '''

    # Get the length of documents
    D = len(texts)

    dict_gs = gs.corpora.Dictionary([[str(i)] for i in range(V)])
    corpus_gs = [dict_gs.doc2bow(text) for text in texts]

    K_lda  = 1*K
    lda_g = gs.models.ldamodel.LdaModel(corpus_gs, id2word=dict_gs, num_topics=K_lda,eval_every = dN_opt,minimum_probability=0.0)    
    
    # Get the topic distribution for each document from the LDA model, 
    # and shuffle the process of the generating process
    p_d_t_lda = np.zeros([D,K])
    tem_save_corpus_dic = {}
    a_tem = list(range(len(corpus_gs)))
    random.shuffle( a_tem )
    for i_d in a_tem:
        one_doc_corpus = corpus_gs[i_d] 
        p_oned_t = lda_g.get_document_topics(one_doc_corpus)    
        for a,b in p_oned_t:
            p_d_t_lda[i_d,a] = b    

    return p_d_t_lda


def find_doc_topic(p_d_t):
    '''
    Classify all documents into different topics according to the probability p(d,t) of different topics for each document. 
    Naive algorithm: choose the topic with the highest probabilit 
    In: 
    - p_d_t : numpy.array([D * K]), the probability of topic t for document d
    Out:
    - list_doc_type : list, (D) , the classification of topic from document 0 to D

    '''


    list_doc_type = []

    for i_d, one_d_t in enumerate(p_d_t):
        a_max = 0
        b_max = 0
        for (a,b) in enumerate(one_d_t):
            if b > b_max:
                a_max = a
                b_max = b

        list_doc_type.append(a_max)

    return list_doc_type




def calculate_nmi(list_doc_type_pp,list_doc_type_lda, K , K_lda = 'None'):
    '''
    Calculate the normalized mutural information for planted classification and the classification based on lda
    In:
    - list_doc_type_pp: list, (# of documents), planted document type 
    - list_doc_type_lda: list, (# of documents), document type based on lda
    - K: # of planted topic
    - K_lda: # of topic inserted into lda model
    Out:
    - NMI_: normalized mutural information
    '''
    if K_lda == 'None':
        K_lda = K


    VI_ = 0.0
    N_ = len(list_doc_type_pp)
    n_tt_ = np.zeros((K,K_lda))

    n_tt_ = np.zeros((K,K_lda))
    list_typepp_typelda = [(list_doc_type_pp[i],list_doc_type_lda[i]) for i in range(N_)]
    c_z1_z2_ = Counter(list_typepp_typelda)

    for z1_z2_,n_z1_z2_ in c_z1_z2_.items():
        n_tt_[z1_z2_[0],z1_z2_[1]] += n_z1_z2_

    p_tt_ = n_tt_/float(N_)
    p_t1_ = np.sum(p_tt_,axis=1)
    p_t2_ = np.sum(p_tt_,axis=0)

    H1_ = sum([-p_*np.log(p_) for p_ in p_t1_ if p_>0.0])
    H2_ = sum([-p_*np.log(p_) for p_ in p_t2_ if p_>0.0])
    MI_ = 0.0
    for i_ in range(K):
        for j_ in range(K_lda):
            p1_ = p_t1_[i_]
            p2_ = p_t2_[j_]
            p12_=p_tt_[i_,j_]
            if p12_> 0.0:
                MI_+=p12_*np.log(p12_/float(p1_*p2_))

    if (H1_+H2_) == 0:
        NMI_ = 0
    else:
        NMI_ = 2.0*MI_/float(H1_+H2_)

    return NMI_




def calculate_accuracy(list_doc_type_pp, list_doc_type_lda, K):
    '''
    Calculate the accuracy of the classification.
    The topic type of a specific document is determined by the naive algorithm.
    The accuracy is obtained by the Kuhn-Munkres algorithm.
    In:
    - list_doc_type_pp: list, (D), the list of topic type for each document by planted
    - list_doc_type_lda: list, (D), the list of topic type for each document by lda
    - K: # of topic
    Out:
    - accurary, the accurary of the classificay 
    '''

    # D : # of documents; K : # of topics
    D = len(list_doc_type_lda)




    # Get the cost matrix of the bipartite graph

    c_array = np.zeros([K,K])
    for pp_k in range(K):
            tem_top_lda_list = []
            for doc in range(D) :        
                if pp_k == list_doc_type_pp[doc] :
                    tem_top_lda_list.append(list_doc_type_lda[doc])            

            # Count the number of documents with different topics
            count_tem = Counter(tem_top_lda_list)

            
            for lda_k in range(K):
                if lda_k in count_tem.keys():
                    c_array[pp_k,lda_k] = - count_tem[lda_k]


    # Solve the assignment problem 
    row_ind, col_ind = optimize.linear_sum_assignment(c_array)
    accuracy = - c_array[row_ind, col_ind].sum()/float(D)

    return accuracy

### Martins revised methods
### clean up above!
def predict_topic_p_td_unsup(p_dt):
	'''predict topic of document based on maximum in topic-doc distribution
	IN: p_d_t: doc-topic distribution D x K
	OUT: list of predicted topics len(D) with entries \in {0,1,...,K-1} 
	'''
	list_doc_topic = []
	D = len(p_dt[:,0])
	for i_d in range(D):
		t = np.argmax(p_dt[i_d])
		list_doc_topic += [t]
	return list_doc_topic


def predict_topic_p_td_sup_knn(p_dt, list_t_d, n_folds = 10, n_neighbors = 10,weights = 'uniform' ):
	'''predict topic of document based on supervised classification.
	   We use k-nearest-neighbor algorithm.

	IN: p_d_t: doc-topic distribution D x K
		list_t_d: list of true topic-labels
		n_folds: division into training and test set (default 10 = 10 nonoverlapping dividisions into 90% training and 10%test)
		n_neighbors: number of neighbors for k-nearest-neighbors algorithm
		weights: another parameter for kNN
	OUT: list of predicted topics len(D) with entries \in {0,1,...,K-1} 
	'''
	D = len(p_dt[:,0])
	list_doc_topic = np.zeros(D)

	### Supervised classification algorithm
	kf = KFold(D, n_folds = n_folds , shuffle = True) #,random_state = 5 )
	# Generate the K-nearest model
	clf = neighbors.KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)            


	for train, test in kf:
		# Get the input and output for trainning 
		p_dt_train = p_dt[train]
		t_train = [list_t_d[ind_train] for ind_train in train]

		# Train the K-nearest model
		clf.fit(p_dt_train, t_train)

		# Get the input data for test, predict the output based on the trained K-nearest model
		p_dt_test = p_dt[test]
		t_test = clf.predict(p_dt_test)

		list_doc_topic[test] = t_test
		# break

	list_doc_topic = list(list_doc_topic.astype('int'))

	return list_doc_topic
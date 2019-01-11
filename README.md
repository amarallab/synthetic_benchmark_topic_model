# Tutorial for synthetic benchmark for topic modeling algorithms

by Hanyu Shi & Martin Gerlach

---
## Introduction

This coding project servers as a brief tutorial about how to use an objective synthetic benchmark to evaluate the performance of different topic modeling algorithms.

More detailed experiment and results can be found in our work, 
**"Hanyu Shi, Martin Gerlach, Isabel Diersen, Doug Downey, Luis A. N. Amaral. 
A new evaluation framework for topic modeling algorithms based on synthetic corpora. 
Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS) 2019"**.

---
## Organization of folders & files

A multilevel list is given below to represent the detailed structure of folders & files in this tutorial. **Folders** are in **bold** while files are not. 
A short description is also given following each item.
Python version used in this project is **3.5.0**.

1. **data**: This folder includes raw dataset.
	1.  **real_corpora**: A set of real-world corpora (collections of documents) should be put into this folder. Note that in this tutorial, only an English corpus is included due to the size limit of the online repo.
		1. reuters_filter_10largest_dic.json: The top 10 categories from [Reuters-21578 dataset](http://www.daviddlewis.com/resources/testcollections/reuters21578/). 
	1.  **stopword_list**: A list of stopwords for different languages should be placed in this folder.
		1. stopword_list_en: An stopword list for English corpora obtained from [MALLET](http://mallet.cs.umass.edu/) package. 
1. **notebook**: This folder contains all the Jupyter notebooks for experiments.
1. **src**: This includes all the scripts
	1. **external**: This contains packages for the external topic modeling algorithms used in this work. However, this is provided in a compressed format.
		1. [hdp-bleilab](https://github.com/blei-lab/hdp).
		1. [mallet-2.0.8RC3](http://mallet.cs.umass.edu/download.php).
		1. [topicmapping](https://amaral.northwestern.edu/resources/software/topic-mapping).
	1. **save_external.zip**: The compressed file for **external**.
	1. **common**: scripts called frequently by multiple other packages.
	1. **corpora**: package to generate synthetic benchmark corpora and call real-world corpora.
	1. **measures**: package to measure the performance of topic modeling algorithms.
	1. **models**: package to facilitate the use of different topic modeling algorithms.
1. requirements.txt: all the python packages used in the experiment.


--- 

## Package installation

- mallet from http://mallet.cs.umass.edu/
	- go to main folder and simply type 'ant', it should say: 'BUILD SUCCESFUL'
	- note that we use the gensim-wrapper to call mallet, which adapted the wrapper/ldamallet.py class for some advanced features. 

- hdp-package from David Blei from https://github.com/blei-lab/hdp
	- install: simply type make in a shell.
	- it uses the hdp-faster algorithm
	- note that we introduced a function "void HDP::save_state_labels" (in state.cpp) adapted from the earlier hdp-package which allows me to get the labels of the individual tokens. the function is called in Main.cpp


- topicmapping from Andrea Lancichinetti from https://bitbucket.org/andrealanci/topicmapping
	- install via python compile.py (as described in readme file of the package)

- nltk package from http://www.nltk.org/
	- You need to install nltk package and the nltk data:
	- 1. For nltk package:
		pip install nltk
	- 2. To download nltk data, run the following lines in the notebook, and in the new window install (double click)  'book'.
		import nltk
		nltk.download()

---
**END**








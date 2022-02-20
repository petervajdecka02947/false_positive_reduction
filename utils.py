#!pip install -U sentence-transformers
#!pip install torch torchvision torchaudio
#pip install sentence-splitter
#!pip install seaborn
# pip install numpy
# sentence-transformers==1.0.4, torch==1.7.0.

import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sentence_transformers import SentenceTransformer, SentencesDataset
from sentence_transformers.losses import TripletLoss
from sentence_transformers.readers import InputExample
from sentence_transformers.evaluation import TripletEvaluator
from tqdm.auto import tqdm
import pickle
import itertools
from sklearn import metrics
import operator
import warnings
from sentence_transformers import models, losses
import time
from datetime import datetime
import random
from collections import defaultdict
from torch.utils.data import DataLoader
import time
import torch, gc
import math
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
    
def get_triple_from_df(df, idx_col ,
                       label_col, text_col, 
                       language_col):
    """
    Function to to create triples from main text column by target label
    """

   
    triplets = []
    idxs = df[idx_col].to_list()
    labels = df[label_col].to_list()
    sentences = df[text_col].to_list()
    languages = df[language_col].to_list()
     
    for index, sentence in enumerate(sentences):
        label = labels[index]
        position = idxs[index]
        language = languages[index]
        
        sentences_in = df[(df[label_col] == label) & (df[idx_col] != position) & (df[language_col] != language)][text_col].to_list()
        
        if len(sentences_in) == 0:
            continue
        sentences_out = df[(df[label_col] != label) & (df[language_col] != language)][text_col].to_list()
        
        positive = random.choice(sentences_in)
        negative = random.choice(sentences_out)
        triplets.append(InputExample(texts = [sentence, positive, negative]))
        #print([sentence, positive, negative])

    return triplets 

def embeddings_sentence_bert(text, IsBase, Bert_name):
    """
    
    """
    
        start = time.time()
        if IsBase==True:            
            model = SentenceTransformer(Bert_name, device = 'cpu')  # model  bert-base-uncased           
        else:     
                     
            word_embedding_model = models.Transformer(Bert_name)
            
            # Apply mean pooling to get one fixed sized sentence vector
            
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)
            
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device = 'cpu')

        
        #Sentences are encoded by calling model.encode()
        sentence_vectors = model.encode(text, show_progress_bar=True, batch_size = 1000)            

        end = time.time()

        print("Time for creating "+ str(len(sentence_vectors))+" embedding vectors " + str((end - start)/60))
        print('Model used :'+ Bert_name )

        return sentence_vectors

def plot_confusion_matrix(cm,
                          title='Confusion matrix',
                          cmap=plt.cm.RdPu):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    lab = ["relevant", "not relevant"]
    classes = np.array(lab)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.0f' # if normalize else 'd'
    fns=11
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j],fmt),
                 horizontalalignment="center",
                 fontsize=fns,
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
   # plt.tight_layout()
    plt.ylim(1.5, -0.5)  # top to bottom solution
    
    return 

def plot_P_R_curve(prec, rec, auc):
    """
    This function plots precision recall curve by having precision recall and auc score as input
    """

    # plot the model precision-recall curve
    plt.figure().set_size_inches(4, 4)
    plt.plot(rec, prec, label = 'Best clasiffier (AUC = %0.2f)' % auc)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()
    
    return 

def train_evaluate_model(data_path):
    """
    Function to split data, train model using grid search and evaluating best model perrformance
    """
    
    start = time.time()
    result=[]
    #pipeline parameters, we use just little for faster grid searching
    parameters = \
        [ \

         {
                'clf': [SVC()],
                'clf__kernel': ['linear', 'poly','sigmoid'],
                'clf__class_weight': ['balanced'],
                'clf__probability': [True]
            },

            {
                'clf': [LogisticRegression()],
                'clf__penalty' : ['l2', 'l1'],
                #'clf__solver':['saga']


            },

            {
                'clf': [RandomForestClassifier()],
            }
        ]
    
    ##################################################################################################################################
    
    data_concat = pd.read_pickle(data_path)
    
    X = data_concat["maintext_embed"].to_list()
    Y = data_concat["target"]
   
    # split with random state
    X_train, X_test, y_train, y_test = train_test_split(X, Y, 
                                                        test_size = 0.3, 
                                                        random_state = 42, 
                                                        stratify = data_concat[["target","language"]])
    
    print("Training starts...")
    
    for params in parameters:
        
        #classifier
        clf = params['clf'][0]

        #getting arguments by
        #popping out classifier
        params.pop('clf')

        #pipeline
        steps = [('clf',clf)]

        #cross validation using
        #Grid Search
        grid = GridSearchCV(Pipeline(steps), param_grid=params, cv=5)
        grid.fit(X_train, y_train)

        #storing result
        result.append\
        (
            {
                'grid': grid,
                'classifier': grid.best_estimator_,
                'best score': grid.best_score_,
                'best params': grid.best_params_,
                'cv': grid.cv
            }
        )
    end = time.time()
    print("Time for training and grid search optimization: " + str((end - start)/60))


    #sorting result by best score
    result = sorted(result, key=operator.itemgetter('best score'),reverse=True)
    # best classifier
    grid = result[0]['grid']
    best_clf = grid.best_estimator_
    
    # Predict labels
    print('Model accuracy on train set is',best_clf.score(X_train, y_train))
    print('Model accuracy on test set is',best_clf.score(X_test, y_test))
    
    y_predicted = best_clf.predict(X_test)
    preds = best_clf.predict_proba(X_test)[:,1]
    
    
    # Evaluation
    matrix = confusion_matrix(y_test, y_predicted) 
    report = metrics.classification_report(y_predicted, y_test)
    precision, recall, threshold = precision_recall_curve(y_test, preds)
    auc_score = metrics.auc(recall, precision)
    
    return result, matrix, report, precision, recall, auc_score
# false_positive_reduction
 - this small project tends to predict and remove false-positives with high probability
 - I do not use lemmas and selected BERT textual representation:
                       * since I have only 3K records and do not expect to extract alignments related to false_positives (in this case I could try logistic regression with tf-idf and using students t-test to extract statisticaly significant n-grams or just unigrams)
                        *  and data are multilingual and I need just one model not to distinguish beetween languages 
- I am also trying to use only CPU to fine-tune SIAMESE Bert with comparison Base SIAMSE Bert Network (https://arxiv.org/abs/1908.10084)
- model also use grid search, but only methods with default params are compared just to save a time
- fine-tunned model outperforms baseline model and could definitely help in detecting relevant false positives texts

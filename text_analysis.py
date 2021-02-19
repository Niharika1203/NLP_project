from transformers import pipeline
from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import pandas as pd
import numpy as np

print("Accessing the data...")
pandas_dataframe = pd.read_csv("training_data.csv")
print(pandas_dataframe.columns)
print("Data Processing...")

tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
model = AutoModel.from_pretrained('deepset/sentence_bert')

# sentence = 'Who are you voting for in 2020?'
# labels = ['body', 'mind', 'consciousness','intelligence',  ]
labels_categories = { 'intelligence' : ['intelligence', 'brilliance','skill','ability', 'sharp', 'knowledge', 'success', 'learn'] ,
            'mind' : ['mind', 'fear' ,'failure','negative', 'doubt' ] ,
            'body' : ['body' , 'health' , 'sense' , 'disease' , 'exercise', 'diet'] ,
            'soul' : ['soul' , 'consciousness' , 'leader' , 'cognizant' ,'serve', 'spirited'] }

labels = ['intelligence', 'brilliance','skill','ability', 'sharp', 'knowledge','mind', 'fear' ,'failure','negative', 'doubt' , 'body' , 'health' , 'sense' , 'disease' , 'exercise', 'diet' , 'Soul' , 'consciousness' , 'leader' , 'cognizant' ,'serve', 'spirited', 'success', 'learn']
sentences = list(pandas_dataframe['Data'])
print(len(sentences))
true_labels = list(pandas_dataframe['Type'])
# run inputs through model and mean-pool over the sequence
# dimension to get sequence-level representations
accuracy = 0

for i in range(len(sentences)) :
    print("Sentence : " , sentences[i] )
    print("True Label : " + str(true_labels[i]))
    inputs = tokenizer.batch_encode_plus([sentences[i]] + labels,
                                         return_tensors='pt',
                                         pad_to_max_length=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output[:1].mean(dim=1)
    label_reps = output[1:].mean(dim=1)

    # now find the labels with the highest cosine similarities to
    # the sentence
    similarities = F.cosine_similarity(sentence_rep, label_reps)
    closest = similarities.argsort(descending=True)
    # for ind in closest:
    #     print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')

    print(f' Similarity label 1: {labels[closest[0]]} \t similarity value : {similarities[closest[0]]}')
    print(f' Similarity label 2: {labels[closest[1]]} \t similarity value : {similarities[closest[1]]}')

    predicted_label = labels[closest[0]]
    predicted_label_2 = labels[closest[1]]
    if predicted_label in labels_categories[true_labels[i].strip().lower()]  or predicted_label_2 in labels_categories[true_labels[i].strip().lower()]  :
        accuracy += 1

print('accuracy Score : {} '.format(accuracy / 120) )
# tokenizer = AutoTokenizer.from_pretrained('deepset/sentence_bert')
# model = AutoModel.from_pretrained('deepset/sentence_bert')
#
# labels = ['body', 'mind', 'consciousness','intelligence' ]#  'soul', ] 'senses',
# sentences = list(pandas_dataframe['Data'])
# print(len(sentences))
# labels_tags = list(pandas_dataframe['Type'])
#
# accuracy = 0
# # run inputs through model and mean-pool over the sequence
# print(sentences[0], labels_tags[0])
# #dimension to get sequence-level representations
# for i in range(len(sentences)) :
#     # sentence = ' I do regular exercising in gym and have strong muscles too. Still I am unable to walk for more than 30 minutes.'
#     inputs = tokenizer.batch_encode_plus([sentences[i]] + labels, return_tensors='pt', pad_to_max_length=True)
#     print(sentences[i])
#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#     output = model(input_ids, attention_mask = attention_mask)[0]
#     sentence_rep = output[:1].mean(dim=1)
#     label_reps = output[1:].mean(dim=1)
#     # now find the labels with the highest cosine similarities to
#     # the sentence
#     similarities = F.cosine_similarity(sentence_rep, label_reps)
#     closest = similarities.argsort(descending=True)
#     print(labels_tags[i])
#     x = 0
#     for ind in closest:
#         # print(ind)
#         if x == 0 or x == 1 :
#             x += 1
#             if labels_tags[i].strip().lower() in [ "body" , "senses" ] :
#                 if str(labels[ind]).strip().lower() in ["body" , "senses"] :
#                     accuracy += 1
#             elif
#             if str(labels[ind]).strip().lower() == labels_tags[i].strip().lower() :
#                 accuracy += 1
#
#         print(f'label: {labels[ind]} \t similarity: {similarities[ind]}')
#     # print(type(closest))
#     # print(f'label: {labels[closest[0]]} \t similarity: {similarities[closest[0]]}')
#
# print(accuracy/120)
# print(labels[i])
# print(len(closest))
# i = 0
# for ind in closest:
    # print(labels_tags[i])
    # i += 1
    # print(ind)


# #!/usr/bin/env python
# # coding: utf-8
#
#
# import math
# import string
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.model_selection import train_test_split
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics import r2_score
# from sklearn import metrics
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import MultiLabelBinarizer
#
#
# # Data import
# print("Accessing the data...")
# pandas_dataframe = pd.read_csv("training_data.csv")
#
# print(pandas_dataframe.columns)
#
# print("Data Processing...")
#
# # Test train split
# sentences_train, sentences_test, y_train, y_test = train_test_split(pandas_dataframe["Data"].values,pandas_dataframe["Type"].values, test_size=0.20, random_state=42)
#
# multilabel_binarizer = MultiLabelBinarizer()
# multilabel_binarizer.fit(y_train)
# Y = multilabel_binarizer.transform(y_train)
#
# count_vect = CountVectorizer()
# X_counts = count_vect.fit_transform(sentences_train)
#
# tfidf_transformer = TfidfTransformer()
# X_tfidf = tfidf_transformer.fit_transform(X_counts)
#
#
# #
# # vectorizer = CountVectorizer()
# # vectorizer.fit(sentences_train)
# # X_train = vectorizer.transform(sentences_train)
# # X_test  = vectorizer.transform(sentences_test)
# # tfidf_transformer = TfidfTransformer()
# # X_train_tfidf = tfidf_transformer.fit_transform(X_train)
# # X_new_tfidf = tfidf_transformer.transform(X_test)
# # print(X_train_tfidf.shape)
# #
# # classifier = LogisticRegression()
# # classifier.fit(X_train_tfidf, y_train)
# # predicted = classifier.predict(X_new_tfidf)
# # score = classifier.score(X_new_tfidf, y_test)
# # print("classes: " , classifier.classes_ )
# # print("Accuracy:", score)
# #
# # from skmultilearn.problem_transform import ClassifierChain
# # from sklearn.linear_model import LogisticRegression
# # # initialize classifier chains multi-label classifier
# # classifier = ClassifierChain(LogisticRegression())
# # # Training logistic regression model on train data
# # classifier.fit(X_train_tfidf, y_train)
# # # predict
# # predictions = classifier.predict(X_new_tfidf)
# # # accuracy
# # score = classifier.score(X_new_tfidf, y_test)
# # print("Accuracy = ",score)  #accuracy_score(y_test,predictions))
# # print("\n")

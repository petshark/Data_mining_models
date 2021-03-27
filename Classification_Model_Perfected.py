# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 12:56:02 2021

@author: carlo
"""
def Process_data(input_parameters):
    import pandas as pd
    from textblob import TextBlob
    
    
    df = pd.read_csv (r'C:\Users\carlo\OneDrive\Escritorio\Holy Hack Data\IMDB-Movie-Data.csv')

    
    category = input_parameters['Category_to_be_predicted']

    if category=='Not Good':
        low_bound, upper_bound = 0, 6.5
    elif category=='Good':
        low_bound, upper_bound = 6.5, 7.5
    elif category=='Very Good':
        low_bound, upper_bound = 7.5, 8.5
    elif category=='Extremely Good':
        low_bound, upper_bound = 8.5, 9   
    
    Dataset_cleaned = pd.DataFrame([])
    
    Observation_cleaned = {}
    
    if input_parameters['Director'] != '':
     
        Ratings = pd.DataFrame([], columns = ['Ratings'])
        Ratings["Director"] = df["Director"]
        Ratings["Ratings"] = ((df["Rating"]>=low_bound)&(df["Rating"]<=upper_bound))*1

        
        ## create director score variable and add actor to dataset

        Director_Scores = Ratings.groupby(["Director"]).sum()/Ratings.groupby(["Director"]).count()



        Director_ratings = {'Director_score':[]};
        for director in df['Director']:
            k = Director_Scores[Director_Scores.index==director]
            score = k.values[0][0]
            Director_ratings['Director_score'].append(score)
            if director == input_parameters['Director']:
               Observation_cleaned['Director_score'] = score
            
    
        Dataset_cleaned.insert(0, 'Director_ratings', Director_ratings['Director_score'], True)
    
    
    if input_parameters['Duration'] != 0:
       ## we add duration to dataset
       Dataset_cleaned['Duration'] = df["Runtime (Minutes)"]
       Observation_cleaned['Duration'] = input_parameters['Duration']

    if input_parameters['Description'] != '':
    ## we add description sentiment ranking to dataset

        

        Description_sentiment_scores = {'Score':[]}

        for description in df['Description']:
            data=description
            check_sentiment = TextBlob(str(data))
            Description_sentiment_scores['Score'].append(check_sentiment.sentiment.polarity)
            
        Dataset_cleaned['Description_sentiment_score'] = pd.DataFrame(Description_sentiment_scores)
        Observation_cleaned['Description_sentiment_score'] = TextBlob(input_parameters['Description']).sentiment.polarity
        
        
    if input_parameters['Genre'] != '':
    ## we add description sentiment ranking to dataset


        Grenre_sentiment_scores = {'Score':[]}

        for genre in df['Genre']:
            data = genre.replace(',',' and ')
            check_sentiment = TextBlob(str(data))
            Grenre_sentiment_scores ['Score'].append(check_sentiment.sentiment.polarity)
    
        Dataset_cleaned['Genre_sentiment_score'] = pd.DataFrame(Grenre_sentiment_scores)
        Observation_cleaned['Genre_sentiment_score'] = TextBlob(input_parameters['Genre']).sentiment.polarity

    ## we add different the target variable
    Dataset_cleaned['Target_Variable'] = Ratings["Ratings"]
    
    
    
    
    return Dataset_cleaned, Observation_cleaned

def Partition_Overall_Dataset(Overall_Dataset):
    ## I divide the dataset into trainig and test sets

    partition = 1/3
    training_set =  Overall_Dataset.sample(frac = partition)
    test_set = Overall_Dataset.drop(training_set.index)

    ## I sepate predictors from target variable for the training set

    target_column_train = training_set['Target_Variable']

    predictors_training_set  = training_set[[i for i in list(training_set.columns) if i != 'Target_Variable']]

    X_train = predictors_training_set.values
    Y_train = target_column_train.values

    ## I sepate predictors from target variable for the test set

    target_column_test = test_set['Target_Variable']

    predictors_test_set  = test_set[[i for i in list(test_set.columns) if i != 'Target_Variable']]

    X_test = predictors_test_set.values
    Y_test = target_column_test.values

    return X_train, Y_train, X_test, Y_test


def Train_Ensemble_Model(X_train, Y_train, X_test, Y_test):
    from sklearn import model_selection
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import roc_auc_score
    from math import sqrt
    import numpy as np


    from sklearn.ensemble import RandomForestClassifier

    #Create a Gaussian Classifier
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, Y_train)
    print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(roc_auc_score(Y_train, clf.predict_proba(X_train)[:, 1])))
    print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(roc_auc_score(Y_test, clf.predict_proba(X_test)[:, 1])))
    return clf

def Make_Prediction(model, observation):
    import pandas as pd
    X_observation = pd.DataFrame.from_dict(observation,orient='index').T
    X_observation = X_observation.values
    prediction = model.predict_proba(X_observation)[:, 1]
    return prediction



def Prediction_Controller(input_parameters):
    data_set_ready, observation_ready = Process_data(input_parameters)
    X_train, Y_train, X_test, Y_test = Partition_Overall_Dataset(data_set_ready)
    prediction_model = Train_Ensemble_Model(X_train, Y_train, X_test, Y_test)
    prediction = Make_Prediction(prediction_model, observation_ready)
    return prediction
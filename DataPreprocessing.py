# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:35:09 2021

@author: carlo
"""




import pandas as pd


df = pd.read_csv (r'C:\Users\carlo\OneDrive\Escritorio\Holy Hack Data\IMDB-Movie-Data.csv')
print (df)




Ratings = pd.DataFrame([], columns = ['Ratings'])
Ratings["Director"] = df["Director"]
Ratings["Ratings"] = (df["Rating"]>=7)*1




## we add year to dataset

Dataset_cleaned = pd.get_dummies(df['Year'], prefix = ["Year"])


## create director score variable and add actor to dataset

Director_Scores = Ratings.groupby(["Director"]).sum()/Ratings.groupby(["Director"]).count()



Director_ratings = {'Director_score':[]};
for director in df['Director']:
    k = Director_Scores[Director_Scores.index==director]
    score = k.values[0][0]
    Director_ratings['Director_score'].append(score)
    


Dataset_cleaned['Director_ratings'] = pd.DataFrame(Director_ratings);

## we add duration to dataset
Dataset_cleaned['Duration'] = df["Runtime (Minutes)"]




## create actor variable


#Actor_list_per_movie = {'Rank':[],'Actor':[]}
#for rank in df["Rank"]:
#    list_of_actors_in_movie_raw = 

#Actor_list_per_movie_raw = pd.DataFrame(df["Actors"].str.split(',').tolist(), index=df["Rank"]).stack()
#Actor_list_per_movie_raw = Actor_list_per_movie.index.drop(['Index']) # var1 variable is currently labeled 0








## we add different the target variable
Dataset_cleaned['Rating_dummy'] = Ratings["Ratings"]

## I divide the dataset into trainig and test sets

partition = 1/3
training_set =  Dataset_cleaned.sample(frac = partition)
test_set = Dataset_cleaned.drop(training_set.index)

## I sepate predictors from target variable for the training set

target_column_train = training_set['Rating_dummy']

predictors_training_set  = training_set[[i for i in list(training_set.columns) if i != 'Rating_dummy']]

X_train = predictors_training_set.values
Y_train = target_column_train.values

print(X_train.shape)
print(Y_train.shape)

## I sepate predictors from target variable for the test set

target_column_test = test_set['Rating_dummy']

predictors_test_set  = test_set[[i for i in list(test_set.columns) if i != 'Rating_dummy']]

X_test = predictors_test_set.values
Y_test = target_column_test.values

print(X_test.shape)
print(Y_test.shape)





        

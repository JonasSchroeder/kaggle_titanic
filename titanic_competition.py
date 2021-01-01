#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:18:25 2020

Kaggle Titanic Competition

@author: jonasschroeder
"""

import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

# change working directory
path = r"/Users/jonasschroeder/OneDrive - MTP-Marketing zwischen Theorie und Praxis e. V/Python/Kaggle/Titanic Competition"
os.chdir(path)

# Import and explore data
'''
# import data
'PassengerId'->
'Survived'-> 0: no , 1: yes
'Pclass'->social economic class (1: upper, 3: lower)
'Name'->
'Sex'->
'Age'->fractional when estimated
'SibSp'->siblings or spouse
'Parch'->parent or child
'Ticket'->ticket number
'Fare'->
'Cabin'->
'Embarked'->port entered the ship from; C = Cherbourg, Q = Queenstown, S = Southampton
Titanic Path: Southampton -> Cherbourg -> Queenstown
 '''
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

train_df.info()

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Data Transformation
#------------------------------------------------------------------------------------------------------------------------------------------------------

# TRAINING DATA

# Rename class
train_df["Pclass"].replace(1, "Upper", inplace=True)
train_df["Pclass"].replace(2, "Middle", inplace=True)
train_df["Pclass"].replace(3, "Lower", inplace=True)

# Replace missing age with median
train_df["Age"].fillna(np.nanmedian(train_df["Age"]), inplace=True)

# Replace NA for embarked with "Unknown"
train_df["Embarked"].fillna("Unknown", inplace=True)

# Replace NA for Cabin with "Unknown"
train_df["Cabin"].fillna("Unknown", inplace=True)

# Extract Name Title
train_df["Title"] = train_df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())
title_list = ["Mr", "Miss", "Mrs", "Master", "Dr", "Rev", "Col"]
train_df.loc[~train_df["Title"].isin(title_list), "Title"] = "NA"

# Extract deck from Cabin
for i in range(0, len(train_df)):
    train_df.at[i, "Deck"] = " ".join(re.findall("[a-zA-Z]+", train_df.at[i, "Cabin"]))

train_df["Deck"].replace("B B", "B", inplace=True)
train_df["Deck"].replace("B B B", "B", inplace=True)
train_df["Deck"].replace("B B B B", "B", inplace=True)
train_df["Deck"].replace("C C", "C", inplace=True)
train_df["Deck"].replace("D D", "D", inplace=True)
train_df["Deck"].replace("C C C", "C", inplace=True)
train_df["Deck"].replace("F G", "F", inplace=True)
train_df["Deck"].replace("F E", "E", inplace=True)


# TEST DATA

# Rename class
test_df["Pclass"].replace(1, "Upper", inplace=True)
test_df["Pclass"].replace(2, "Middle", inplace=True)
test_df["Pclass"].replace(3, "Lower", inplace=True)

# Replace missing age with median
test_df["Age"].fillna(np.nanmedian(test_df["Age"]), inplace=True)

# Replace missing fare with median
test_df["Fare"].fillna(np.nanmedian(test_df["Fare"]), inplace=True)

# Replace NA for embarked with "Unknown"
test_df["Embarked"].fillna("Unknown", inplace=True)

# Replace NA for Cabin with "Unknown"
test_df["Cabin"].fillna("Unknown", inplace=True)

# Extract Name Title
test_df["Title"] = test_df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())
title_list = ["Mr", "Miss", "Mrs", "Master", "Dr", "Rev", "Col"]
test_df.loc[~test_df["Title"].isin(title_list), "Title"] = "NA"

# Extract deck from Cabin
for i in range(0, len(test_df)):
    test_df.at[i, "Deck"] = " ".join(re.findall("[a-zA-Z]+", test_df.at[i, "Cabin"]))

test_df["Deck"].replace("B B", "B", inplace=True)
test_df["Deck"].replace("B B B", "B", inplace=True)
test_df["Deck"].replace("B B B B", "B", inplace=True)
test_df["Deck"].replace("C C", "C", inplace=True)
test_df["Deck"].replace("E E", "E", inplace=True)
test_df["Deck"].replace("D D", "D", inplace=True)
test_df["Deck"].replace("C C C", "C", inplace=True)
test_df["Deck"].replace("F G", "F", inplace=True)
test_df["Deck"].replace("F E", "E", inplace=True)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# EDA
#------------------------------------------------------------------------------------------------------------------------------------------------------

# column names
train_df.columns

# overview of data
train_df.info()

# summary statistics
sum_stats = train_df.describe()

# correlation between all columns
matrix = np.triu(train_df.corr())
sns.heatmap(train_df.corr(), annot=True, fmt=".1g", vmin=-1, vmax=1, center=0, cmap="coolwarm", square=True, mask=matrix)

# Survived
count_survived = train_df["Survived"].value_counts()
count_survived .plot.bar()
plt.title("Number of Survivors")


# Number per class
count_class = train_df["Pclass"].value_counts()
count_class.plot.bar()
plt.title("Number per Class")

# Number per Title
count_title = train_df["Title"].value_counts()
count_title.plot.bar()
plt.title("Number per Title")

# Number of siblings or spouses
count_sibsp = train_df["SibSp"].value_counts()
count_sibsp.plot.bar()
plt.title("Number of Siblings or Spouses")

# Number per port
count_port = train_df["Embarked"].value_counts()
count_port.plot.bar()
plt.title("Number per Port")

# Number per deck
count_deck = train_df["Deck"].value_counts()
count_deck.plot.bar()
plt.title("Number per Deck")

# Age Histogram -> how to deal with age na (177)?
train_df["Age"].plot.hist()

# Average fare per class
train_df.groupby("Pclass").agg({"Fare":"mean"})

# compare survived vs non-survived
sns.displot(train_df, x="Fare", hue="Survived")

# Survivor split Class
tbl = pd.pivot_table(train_df, index="Survived", columns="Pclass", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar')
sns.countplot(data=train_df, y="Survived", hue="Pclass", color="Orange" )

# Survivor split Sex
tbl = pd.pivot_table(train_df, index="Survived", columns="Sex", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # female survive
sns.countplot(data=train_df, y="Survived", hue="Sex", color="Orange" )

# Survivor split SibSp
tbl = pd.pivot_table(train_df, index="Survived", columns="SibSp", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar')
sns.countplot(data=train_df, y="Survived", hue="SibSp", color="Orange" )

# Survivor split Title
tbl = pd.pivot_table(train_df, index="Survived", columns="Title", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # -> same as for sex
# sns.countplot(data=train_df, y="Survived", hue="Title", color="Orange" )

# Survivor split Embarked
tbl = pd.pivot_table(train_df, index="Survived", columns="Embarked", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar')
sns.countplot(data=train_df, y="Survived", hue="Embarked", color="Orange" )

# Survivor split Deck
tbl = pd.pivot_table(train_df, index="Survived", columns="Deck", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # unknown deck for vast majority of people?
sns.countplot(data=train_df, y="Survived", hue="Deck", color="Orange" )

# Fare per Deck, Port, Class
fare_per_group = train_df.groupby(["Deck", "Embarked", "Pclass"]).agg({"Fare": "mean"})
train_df.groupby(["Embarked"]).agg({"Fare": "mean"}) # unknown port highest average price - why? reason: all from deck B
train_df.groupby(["Deck"]).agg({"Fare": "mean"}) # B > C > D > E > A > rest
train_df.groupby(["Pclass"]).agg({"Fare": "mean"}) # positive price-class relationship


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Pre-processing: Standardization and One-Hot Encoding
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

train_df.columns
test_df.columns

train_df_pre = train_df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
test_df_pre = test_df.drop(columns=[ "Name", "Ticket", "Cabin"])

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Embarked", "Deck", "Sex", "Title"]

col_transformer = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ],
    remainder="passthrough")

# Fit transform TRAIN
train_array_transformed = col_transformer.fit_transform(train_df_pre)

# Convert numpy.ndarray to pd.DataFrame
train_df_transformed = pd.DataFrame(data=train_array_transformed)
#train_df_transformed = pd.DataFrame(data=train_array_transformed.toarray())

# Rename columns
column_names = num_attribs + list(col_transformer.named_transformers_['cat'].get_feature_names()) + ["Survived"]
train_df_transformed.columns = column_names


# Fit transform TEST
test_array_transformed = col_transformer.fit_transform(test_df_pre)

# Convert numpy.ndarray to pd.DataFrame
test_df_transformed = pd.DataFrame(data=test_array_transformed)

# Rename columns
column_names = num_attribs + list(col_transformer.named_transformers_['cat'].get_feature_names()) + ["PassengerId"]
test_df_transformed.columns = column_names


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Random Forest Classifier 0.81
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

# Random forrest classifier
rnd_clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

cv_rnd_clf = cross_val_score(rnd_clf, X_train, y_train, cv=5)
print(cv_rnd_clf)
print("mean accuracy: " + str(cv_rnd_clf.mean()))

rnd_clf.fit(X_train, y_train)

y_test = rnd_clf.predict(test_df_transformed)

# Feature Importance
feat_importances = pd.Series(rnd_clf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("rnd_clf_simple.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression 0.83
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

#  Logistic regression
lr = LogisticRegression(max_iter=2000)

cv_lr = cross_val_score(lr, X_train, y_train, cv=5)
print(cv_lr)
print("mean accuracy: " + str(cv_lr.mean()))

lr.fit(X_train, y_train)

y_test = lr.predict(test_df_transformed)

# Feature importance
feature_importance_lr = pd.DataFrame()
feature_importance_lr["Features"] = X_train.columns
feature_importance_lr["lr_coeff"] = np.transpose(lr.coef_)
feature_importance_lr["coef_abs"] = np.transpose(np.abs(lr.coef_))

plt.figure().set_size_inches(10, 6)
fg3 = sns.barplot(x='Features', y='lr_coeff',data=feature_importance_lr, palette="Blues_d")
fg3.set_xticklabels(rotation=90, labels=feature_importance_lr.Features)
plt.title("Logistic Regression Model: Regression Coefs")


# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("log_reg_simple.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Naive Bayes 0.75
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

#  Naive Bayes
gnb = GaussianNB()

cv_gnb = cross_val_score(gnb, X_train, y_train, cv=5)
print(cv_gnb)
print("mean accuracy: " + str(cv_gnb.mean()))

gnb.fit(X_train, y_train)

y_test = gnb.predict(test_df_transformed)

# Feature importance

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("nb_simple.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Voting Classifier 0.79
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

#  Voting Classifier
voting_clf = VotingClassifier(estimators=[("rnd_clf", rnd_clf), ("lr", lr), ("gnb", gnb)], voting="soft")

cv_voting = cross_val_score(voting_clf, X_train, y_train, cv=5)
print(cv_voting)
print("mean accuracy: " + str(cv_voting.mean()))

voting_clf.fit(X_train, y_train)

y_test = voting_clf.predict(test_df_transformed)

# Feature importance

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("voting_simple.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Random Forest Classifier TUNED 0.84
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

# Random forrest classifier
rf = RandomForestClassifier()

# Find optimal parameter settings using Randomized Search
# Round 1: Board
param_grid =  {'n_estimators': [100,500,1000], 
               'bootstrap': [True,False],
               'max_depth': [50,75,100,125, 150,None],
               'max_features': ['auto','sqrt'],
               'min_samples_leaf': [1,2,4,10],
               'min_samples_split': [5,10,15]}

rnd_clf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=True, n_jobs=-1)

best_rnd_clf = rnd_clf.fit(X_train, y_train)
print("Best Random Forest Score: " + str(best_rnd_clf .best_score_))
print("Best Parameter:  " + str(best_rnd_clf .best_params_))

# Find optimal parameter settings using Randomized Search
# Round 2: Specified
param_grid =  {'n_estimators': [400,450,500,550], 
               'bootstrap': [False],
               'max_depth': [3,5,10,20,50,75,100,None],
               'max_features': ['auto'],
               'min_samples_leaf': [1,2,4],
               'min_samples_split': [2,5,10]}

rnd_clf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=True, n_jobs=-1)

best_rnd_clf = rnd_clf.fit(X_train, y_train)
print("Best Random Forest Score: " + str(best_rnd_clf .best_score_))
print("Best Parameter:  " + str(best_rnd_clf .best_params_))


y_test = rnd_clf.predict(test_df_transformed)

# Feature Importance
best_rf = best_rnd_clf.best_estimator_.fit(X_train, y_train)
feat_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("rnd_clf_tuned.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Logistic Regression TUNED 0.83 -> no improvement
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

#  Logistic regression
lr = LogisticRegression()

# Find optimal parameter settings using GridSearch
param_grid = {"max_iter" : [2000],
              "penalty" : ["l1", "l2"],
              "C" : np.logspace(-4, 4, 20),
              "solver" : ["liblinear"]}
    
lr_clf = GridSearchCV(lr, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)

best_lr_clf = lr_clf.fit(X_train, y_train)
print("Best Logistic Regression Score: " + str(best_lr_clf.best_score_))
print("Best Parameter:  " + str(best_lr_clf.best_params_))


y_test = lr_clf.predict(test_df_transformed)

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("log_reg_tuned.csv", index=False)

#------------------------------------------------------------------------------------------------------------------------------------------------------
# Voting Classifier TUNED  hard 0.83, weighted soft 0.82
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import VotingClassifier

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

best_lr = best_lr_clf.best_estimator_
best_rf = best_rnd_clf.best_estimator_

# Hard Voting

voting_clf_hard = VotingClassifier(estimators=[("rnd_clf", best_rf), ("lr", best_lr), ("gnb", gnb)], voting = 'hard') 

print('voting_clf_hard :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5))
print('voting_clf_hard mean :',cross_val_score(voting_clf_hard,X_train,y_train,cv=5).mean()) #0.83

voting_clf_hard.fit(X_train, y_train)
y_test = voting_clf_hard.predict(test_df_transformed)

# Soft Voting with optimized weights

voting_clf_soft = VotingClassifier(estimators=[("rnd_clf", best_rf), ("lr", best_lr), ("gnb", gnb)], voting = 'soft') 

params = {'weights' : [[1,1,1],[1,2,1],[1,1,2],[2,1,1],[2,2,1],[1,2,2],[2,1,2]]}
vote_weight = GridSearchCV(voting_clf_soft, param_grid = params, cv = 5, verbose = True, n_jobs = -1)
best_clf_weight = vote_weight.fit(X_train, y_train)
print("Best Weighted Vote Score: " + str(best_clf_weight .best_score_))
print("Best Parameter:  " + str(best_clf_weight .best_params_)) #0.82

y_test = best_clf_weight.best_estimator_.predict(test_df_transformed)


# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("voting_hard_tuned.csv", index=False)
#export_df.to_csv("voting_soft_tuned.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Random Forest TUNED without cabin/deck features 0.839
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

train_df_transformed.columns
X_train = train_df_transformed.drop(columns=["Survived", 'x2_A', 'x2_B', 'x2_C', 'x2_D', 'x2_E', 'x2_F', 'x2_G', 'x2_T', 'x2_Unknown'])
y_train = train_df_transformed["Survived"]

# Random forrest classifier
rf = RandomForestClassifier()

# Find optimal parameter settings using Randomized Search
# Round 1: Board
param_grid =  {'n_estimators': [100,500,1000], 
               'bootstrap': [True,False],
               'criterion': ["entropy", "gini"],
               'max_depth': [50,75,100,125,150,None],
               'max_features': ['auto','sqrt'],
               'min_samples_leaf': [1,2,4,10],
               'min_samples_split': [5,10,15]}

rnd_clf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=True, n_jobs=-1)

best_rnd_clf = rnd_clf.fit(X_train, y_train)
print("Best Random Forest Score: " + str(best_rnd_clf .best_score_))
print("Best Parameter:  " + str(best_rnd_clf .best_params_))

# Find optimal parameter settings using Randomized Search
# Round 2: Specified
param_grid =  {'n_estimators': [25,50,75,100], 
               'bootstrap': [True],
               'criterion': ["entropy", "gini"],
               'max_depth': [50,75,100,125,150,175,200,None],
               'max_features': ['auto'],
               'min_samples_leaf': [1,2,4],
               'min_samples_split': [2,5,10,15]}

rnd_clf = RandomizedSearchCV(rf, param_distributions=param_grid, n_iter=300, cv=5, verbose=True, n_jobs=-1)

best_rnd_clf = rnd_clf.fit(X_train, y_train)
print("Best Random Forest Score: " + str(best_rnd_clf .best_score_))
print("Best Parameter:  " + str(best_rnd_clf .best_params_))

test_df_transformed_short = test_df_transformed.drop(columns=['x2_A', 'x2_B', 'x2_C', 'x2_D', 'x2_E', 'x2_F', 'x2_G', 'x2_Unknown'])
y_test = rnd_clf.predict(test_df_transformed_short)

# Feature Importance
best_rf = best_rnd_clf.best_estimator_.fit(X_train, y_train)
feat_importances = pd.Series(best_rf.feature_importances_, index=X_train.columns)
feat_importances.nlargest(20).plot(kind='barh')

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("rnd_clf_tuned_nocabin.csv", index=False)






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

# Replace missing age with average
train_df["Age"].fillna(np.mean(train_df["Age"]), inplace=True)

# Replace NA for embarked with "Unknown"
train_df["Embarked"].fillna("Unknown", inplace=True)

# Replace NA for Cabin with "Unknown"
train_df["Cabin"].fillna("Unknown", inplace=True)

# Extract Name Title
train_df["Title"] = train_df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

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

# Replace missing age with average
test_df["Age"].fillna(np.mean(test_df["Age"]), inplace=True)

# Replace missing fare with average
test_df["Fare"].fillna(np.mean(test_df["Fare"]), inplace=True)

# Replace NA for embarked with "Unknown"
test_df["Embarked"].fillna("Unknown", inplace=True)

# Replace NA for Cabin with "Unknown"
test_df["Cabin"].fillna("Unknown", inplace=True)

# Extract Name Title
test_df["Title"] = test_df.Name.apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Extract deck from Cabin
for i in range(0, len(test_df)):
    test_df.at[i, "Deck"] = " ".join(re.findall("[a-zA-Z]+", test_df.at[i, "Cabin"]))

test_df["Deck"].replace("B B", "B", inplace=True)
test_df["Deck"].replace("B B B", "B", inplace=True)
test_df["Deck"].replace("B B B B", "B", inplace=True)
test_df["Deck"].replace("C C", "C", inplace=True)
train_df["Deck"].replace("E E", "E", inplace=True)
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

# Survivor split Sex
tbl = pd.pivot_table(train_df, index="Survived", columns="Sex", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # female survive

# Survivor split SibSp
tbl = pd.pivot_table(train_df, index="Survived", columns="SibSp", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar')

# Survivor split Title
tbl = pd.pivot_table(train_df, index="Survived", columns="Title", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # -> same as for sex

# Survivor split Embarked
tbl = pd.pivot_table(train_df, index="Survived", columns="Embarked", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar')

# Survivor split Deck
tbl = pd.pivot_table(train_df, index="Survived", columns="Deck", values="Ticket", aggfunc="count")
ax = tbl.T.plot(kind='bar') # unknown deck for vast majority of people?

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
cat_attribs = ["Pclass", "Embarked", "Deck", "Sex"]

col_transformer = ColumnTransformer([
    ("num", StandardScaler(), num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
    ],
    remainder="passthrough")

# Fit transform TRAIN
train_array_transformed = col_transformer.fit_transform(train_df_pre)

# Convert numpy.ndarray to pd.DataFrame
train_df_transformed = pd.DataFrame(data=train_array_transformed)

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
# Random Forest Classifier 0.80
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
# Logistic Regression 0.81
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

# Export for submit
export_df = pd.DataFrame()
export_df["PassengerId"] = test_df_transformed["PassengerId"].astype(int)
export_df["Survived"] = y_test.astype(int)
export_df.to_csv("log_reg_simple.csv", index=False)


#------------------------------------------------------------------------------------------------------------------------------------------------------
# Naive Bayes 0.72
#------------------------------------------------------------------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

X_train = train_df_transformed.drop(columns=["Survived"])
y_train = train_df_transformed["Survived"]

#  Logistic regression
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



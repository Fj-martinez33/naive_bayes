# Librerias

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from pickle import dumps

data_url = "https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv"

def DataCompiler(url, sep = ","):
    data = pd.read_csv(url)
    data.to_csv("../data/raw/reviews.csv")
    return data

data = DataCompiler(data_url)

data.drop("package_name", axis = 1, inplace = True)

def StripPhase(dataset, col):
        dataset[col] = dataset[col].str.strip().str.lower()
        return dataset

StripPhase(data, "review")

x_train, x_test, y_train, y_test = train_test_split(data["review"], data["polarity"], test_size= 0.2, random_state=42)

def VectorizeString(train, test):
    vec_model = CountVectorizer(stop_words = "english")
    x_train_vect = vec_model.fit_transform(train).toarray().squeeze()
    x_test_vect = vec_model.transform(test).toarray().squeeze()
    return x_train_vect, x_test_vect

x_train_vect, x_test_vect = VectorizeString(x_train, x_test)

t_models = [MultinomialNB(), GaussianNB(), BernoulliNB()]

def TrainingModel(t_model):
    results = []
    models = []

    for i in range(len(t_model)):

        model = t_model[i]
        model.fit(x_train_vect, y_train)
        models.append(model)

        y_test_pred = model.predict(x_test_vect)
        y_train_pred = model.predict(x_train_vect)
        result = {"Index" : str(t_models[i]), "train_score" : accuracy_score(y_train, y_train_pred), "test_score" : accuracy_score(y_test, y_test_pred)}
        results.append(result)

    return results, models

pre_results, pre_models = TrainingModel(t_models)

hyperparameters = {"alpha" : np.linspace(0, 100, 1000), "force_alpha" : [True, False], "fit_prior" : [True, False]}

grid = GridSearchCV(pre_models[0], hyperparameters, scoring="accuracy")

grid.fit(x_train_vect, y_train)

grid.best_params_

clf = grid.best_estimator_

dump(clf, open(f"../models/Multinomialmodel.sav", "wb"))

y_test_pred = clf.predict(x_test_vect)

accuracy_score(y_test, y_test_pred)
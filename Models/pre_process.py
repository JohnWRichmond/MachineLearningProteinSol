from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import pandas as pd

#Function which returns a normalized pandas dataframe
def normalize(Data):
    return (Data - Data.mean())/Data.std()

#Function which ranks features importance according to the boruta algorithm and returns a dataframe where each feature is ranked
def feature_selection(Data, Target):
    forest = RandomForestRegressor(
        n_jobs = -1,
        max_depth = 5)

    boruta = BorutaPy(
        estimator = forest,
        n_estimators = 'auto',
        max_iter = 100)

    boruta.fit(np.array(Data.drop(columns = Target)), np.array(Data[Target]))

    Feature_rankings = pd.DataFrame(index = Data.drop(columns = Target).columns,
                        data = boruta.ranking_,
                        columns = ['Feature_ranking']).sort_values(by = ['Feature_ranking'])

    Feature_rankings['Feature_ranking'] = Feature_rankings['Feature_ranking'].rank(method='dense').astype(int)

    return Feature_rankings

#Function which is used to define which preprocessing and feature selection method to use
def preprocess(data, Target):
    data = normalize(data)

    feature_rankings = feature_selection(data, Target)

    return data, feature_rankings

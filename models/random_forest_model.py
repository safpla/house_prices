
"""
Random Forest Model

"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, cross_val_predict
import pre_data
from sklearn.metrics import mean_squared_error
import numpy as np
from utilities import evaluation
import matplotlib.pyplot as plt

# local
import os, sys
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, root_path)
from models.basic_regressor import Basic_regressor

class Randomforest(Basic_regressor):
    def __init__(self, config=None, exp_name='new_exp',rfr = None):
        self.config = config
        self.exp_name = exp_name
        self.rfr = RandomForestRegressor()

    def train(self, features, response,columns):
        params = {}
        # default para
        self.rfr = RandomForestRegressor(
            oob_score=True,
            random_state=10)
        self.rfr.fit(features,response)
        #print("oob_score", self.rfr.oob_score_)

        # Perform Grid-Search
        param_test1 = {'n_estimators': (10, 50, 100, 1000)}
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param_test1,
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        grid_result = gsc.fit(features, response)
        #print(grid_result.grid_scores_,grid_result.best_params_,grid_result.best_score_)
        params["n_estimators"] = grid_result.best_params_["n_estimators"]

        param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param_test2,
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        grid_result = gsc.fit(features, response)
        #print(grid_result.grid_scores_, grid_result.best_params_, grid_result.best_score_)
        params["max_depth"] = grid_result.best_params_["max_depth"]
        params["min_samples_split"] = grid_result.best_params_["max_depth"]


        param_test3 = { 'min_samples_leaf': range(10, 60, 10)}
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid=param_test3,
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        grid_result = gsc.fit(features, response)
        #print(grid_result.grid_scores_, grid_result.best_params_, grid_result.best_score_)
        params["min_samples_leaf"] = grid_result.best_params_["min_samples_leaf"]

        self.rfr = RandomForestRegressor( n_estimators=params["n_estimators"],
                                                 max_depth = params["max_depth"],
                                                 min_samples_split = params["min_samples_split"],
                                                 min_samples_leaf = params["min_samples_leaf"],
                                                 oob_score = True,
                                                 random_state=False, verbose=False)

        self.rfr.fit(features,response)

        # see the importance of features
        #print(columns)
        importances = self.rfr.feature_importances_
        indices = np.argsort(importances)[::-1]
        #for f in range(features.shape[1]):
        #    print(str(f + 1), columns[indices[f]],importances[indices[f]])

        data = importances
        labels = columns

        # plt.bar(range(len(data)), data, tick_label=labels)
        # plt.show()


        #print("oob_score", self.rfr.oob_score_)



    def predict(self,features):
        predictions = self.rfr.predict(features)
        #print("predictions",predictions)
        return predictions

    def evaluation(self,housedata):
        """
                Input: housedata: instance of class pre_data.Housedata
                """
        k = 9
        performance = []
        columns = housedata.data.columns
        for i in range(k):
            self.train(housedata.train_features[i], housedata.train_targets[i],columns)
            predictions = self.predict(housedata.valid_features[i])
            print(np.shape(housedata.valid_features[i]))
            print(np.shape(predictions))
            targets = housedata.valid_targets[i]
            print(np.shape(targets))
            performance.append(evaluation(predictions, targets))
        print(performance)




if __name__ == "__main__":
    data_HOA,data_LOT = pre_data.pre_data()
    otherinfo = data_HOA.otherinfo
    print(otherinfo)
    X_train = data_HOA.train_features
    y_train = data_HOA.train_targets
    X_test = data_HOA.test_features
    y_test = data_HOA.test_targets

    columns = data_HOA.data.columns
    # data = train_test_split(data_HOA.data, data_HOA.target,
    #                         train_size=0.7, random_state=0)
    # X_train, X_test, y_train, y_test = data

    rfr_model = Randomforest()
    rfr_model.evaluation(data_HOA)



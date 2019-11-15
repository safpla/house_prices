"""
Random Forest Model

"""
import os, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import numpy as np

# import local
root_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, root_path)
import pre_data
from models.basic_regressor import Basic_regressor

class Randomforest(Basic_regressor):
    def __init__(self, config=None, exp_name='new_exp',rfr = None):
        self.config = config
        self.exp_name = exp_name
        self.rfr = RandomForestRegressor()


    def train(self, features, response):
        # Perform Grid-Search
        gsc = GridSearchCV(
            estimator=RandomForestRegressor(),
            param_grid={
                'max_depth': range(3, 7),
                'n_estimators': (10, 50, 100, 1000),
            },
            cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1
        )
        grid_result = gsc.fit(features, response)
        best_params = grid_result.best_params_
        self.rfr = RandomForestRegressor(max_depth=best_params["max_depth"],
                                                 n_estimators=best_params["n_estimators"],
                                                 random_state=False, verbose=False)
        self.rfr.fit(features,response)


    def predict(self,features,response):

        predictions = self.rfr.predict(features)
        print("predictions",predictions)
        # print("response",test_score)
        predictions.reshape(-1,1)
        response.reshape(-1,1)
        #test_score = np.sqrt(-cross_val_score(self.rfr, predictions, response, cv=5, scoring='neg_mean_squared_error'))
        MSE = mean_squared_error(predictions, response);
        #print("test_score",test_score)
        print("MSE",MSE)
        #print(predictions)
        return predictions


if __name__ == "__main__":
    data_HOA,data_LOT = pre_data.pre_data()
    data = train_test_split(data_HOA.data, data_HOA.target,
                            test_size=0.7, random_state=0)
    X_train, X_test, y_train, y_test = data

    rfr_model = Randomforest()
    rfr_model.train(X_train,y_train)
    rfr_model.predict(X_test,y_test)


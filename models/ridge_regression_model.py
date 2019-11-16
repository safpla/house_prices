
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict,GridSearchCV
from basic_regressor import Basic_regressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import pre_data

class grid():

    def __init__(self,model):
        self.model=model

    #Find parameters
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=9, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
        return grid_search.best_params_

class performance():

    #Calculate MeanAbsoluteError,MeanSquaredError,MedianAbsolutePercentageError and MdAPEin5PercentCounts
    def evaluation(predictions, response, metrics=['MAE', 'MSE', 'MdAPE', '5pct']):
        outputs = []
        predictions = np.array(predictions)
        response = np.array(response)
        for metric in metrics:
            if metric == 'MAE':
                output = sum(abs(response - predictions)) / len(response)
                outputs.append(output)
            elif metric == 'MSE':
                output = sum(np.square(response - predictions)) / len(response)
                outputs.append(output)
            elif metric == 'MdAPE':
                p = abs(response - predictions) / response
                outputs.append(np.median(p))
            elif metric == '5pct':
                p = abs(response - predictions) / response
                counts = sum(p<0.05)
                outputs.append(counts / len(response))
        return outputs


class ridge_regression(Basic_regressor):

    def __init__(self, config=None, exp_name='new_exp',ridge = None):
        self.config = config
        self.exp_name = exp_name
        self.ridge = Ridge()


    #train
    def train(self,features,response):
        alphas = np.logspace(-3,2,50)
        best_params = grid(Ridge()).grid_get(features,response,{'alpha': alphas,'max_iter':[10000]})
        self.ridge = Ridge(alpha=best_params['alpha'])
        self.ridge.fit(features,response)

    #predict
    def predict(self,features,response):
        predictions = self.ridge.predict(features)
        print("predictions",predictions)
        predictions.reshape(-1,1)
        response.reshape(-1,1)
        perf = performance.evaluation(predictions,response)
        print("MAE, MSE, MdAPE, 5pct",perf)
        return(predictions)

if __name__ == "__main__":
    data_HOA,data_LOT = pre_data.pre_data()
    data = train_test_split(data_HOA.data, data_HOA.target,
                            test_size=0.1, random_state=0)
    X_train, X_test, y_train, y_test = data
    ridge_model = ridge_regression()
    ridge_model.train(X_train,y_train)
    ridge_model.predict(X_test,y_test)

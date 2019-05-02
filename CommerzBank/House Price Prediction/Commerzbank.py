
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, train_test_split

from xgboost import XGBRegressor, DMatrix
import xgboost

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def tune_xgb_cv(params_untuned,params_sklearn,scoring='roc_auc', n_jobs=4, cv=5,verbose=10):

    for param_untuned in params_untuned:
        print('==========  ', param_untuned, '  ==============')
        print_params(params_sklearn)
        estimator = xgb.XGBClassifier(**params_sklearn)
        # if(param_untuned.keys()[0] == 'n_estimators'):
        #     cv = 1
        grid_search = GridSearchCV(estimator, param_grid=param_untuned, scoring=scoring, n_jobs=n_jobs, cv=cv, verbose=verbose)
        grid_search.fit(x, y)
        df = pd.DataFrame(grid_search.cv_results_)[['params', 'mean_train_score', 'mean_test_score']]
        print(df)
        print('the best_params : ', grid_search.best_params_)
        print('the best_score  : ', grid_search.best_score_)
        for k,v in grid_search.best_params_.items():
            params_sklearn[k] = v
    return estimator,params_sklearn

def hist_columns(dataset, chosen_columns):
    for column in chosen_columns:
        print(column)
        train[column].hist(bins=50)
        plt.title(column)
        plt.show()

if __name__ == '__main__':
    train_columns = ['PropertyID', 'Town', 'Bedrooms', 'Bathroom', 'PropertyType', 'DistanceFromCBD', 'Landsqm', 'floorsqm', 'YearBuilt', 'Region', 'TownDensity', 'SaleDate', 'SalePrice']
    test_columns = ['PropertyID', 'Town', 'Bedrooms', 'Bathroom', 'PropertyType', 'DistanceFromCBD', 'Landsqm', 'floorsqm', 'YearBuilt', 'Region', 'TownDensity', 'SaleDate']
    test = pd.read_csv('test.csv', usecols=test_columns)
    train =pd.read_csv('train.csv', usecols=train_columns)
    train.info()

    # # exploration
    town_summary = train.groupby('Town').median()['SalePrice'].sort_values()
    town_summary.plot()
    plt.show()

    town_density_summary = train.groupby('Town').median()[['SalePrice', 'TownDensity']].sort_values('SalePrice')
    town_density_summary['TownDensity'].plot()
    plt.show()

    propertytype_summary = train.groupby('PropertyType').median()['SalePrice'].sort_values()
    propertytype_summary.plot()
    plt.show()

    region_summary = train.groupby('Region').median()['SalePrice'].sort_values()
    region_summary.plot()
    plt.show()
    
    saledate_summary = train.groupby('SaleDate').median()['SalePrice'].sort_values()
    train['SaleDateDate'] = train['SaleDate'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
    saledatedate_summary = train.groupby('SaleDateDate').median()['SalePrice'].sort_index()
    saledatedate_summary.plot()
    plt.show()   
    # nothing here

    # variable encoding
    town_ref = pd.DataFrame(index=town_summary.index, columns=['Town'], data=list(range(len(town_summary)))).to_dict()
    propertytype_ref = pd.DataFrame(index=propertytype_summary.index, columns=['PropertyType'], data=list(range(len(propertytype_summary)))).to_dict()
    region_ref = pd.DataFrame(index=region_summary.index, columns=['Region'], data=list(range(len(region_summary)))).to_dict()
    encoding = {}
    encoding.update(town_ref)
    encoding.update(propertytype_ref)
    encoding.update(region_ref)
    train.replace(encoding, inplace=True)


    chosen_columns = ['SalePrice', 'Town', 'Bedrooms', 'Bathroom', 'PropertyType', 'DistanceFromCBD', 'Landsqm', 'floorsqm', 'YearBuilt', 'Region']
    #hist_columns(train, chosen_columns)

    ### reashaping features ###
    skwed_features = ['Landsqm', 'floorsqm', 'SalePrice']
    for feature in skwed_features:
        train[feature] = train[feature].apply(np.log1p)
    #hist_columns(train, chosen_columns)

    # getting rid of useless shit
    #train.fillna(0, inplace=True)
    #train.fillna(train.median(), inplace=True)
    #train.fillna(train.mean(), inplace=True)

    selected_train_columns = ['Town', 'Bedrooms', 'Bathroom', 'PropertyType', 'DistanceFromCBD', 'Landsqm', 'floorsqm', 'YearBuilt', 'Region']
    y = train['SalePrice'].to_numpy()
    X = train[selected_train_columns].to_numpy()

    data_dmatrix = DMatrix(data=X,label=y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    #model
    xgb = XGBRegressor()
    xgb = XGBRegressor(learning_rate=0.01, n_estimators=800,
                                     max_depth=10, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.5,
                                     objective='reg:linear',
                                     seed=27,
                                     reg_alpha=0,
                                     reg_lambda=0) # reg_alpha = 0.00006
    xgb.fit(X_train, y_train)
    #model evaluation
    print('XGBoost in sample')
    R2in = r2_score(y_train, xgb.predict(X_train))
    msein = mean_squared_error(y_train, xgb.predict(X_train))
    print('R Squared: ', str(R2in))
    print('mean square error: ', str(msein))
    print()
    print('XGBoost out of sample')
    predictions = xgb.predict(X_test)
    R2out = r2_score(y_test, predictions)
    mseout = mean_squared_error(y_test, predictions)   
    print('R Squared: ', str(R2out))
    print('mean square error: ', str(mseout))
    print()


    # XGBoost cross validation function attempt... but a more detailed grid search is needed in the hyper parameter space
    params = {"objective":"reg:linear",'colsample_bytree': 0.7, 'colsample_bytree':0.5, 'learning_rate': 0.01,
                'max_depth': 10} #, 'alpha': 10}
    cv_results = xgboost.cv(dtrain=data_dmatrix, params=params, nfold=7,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", seed=123) #, as_pandas=True

    xg_reg = xgboost.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

    ###########################################
    ### preparing test data and predictions ###
    ###########################################
    test.replace(encoding, inplace=True)
    X_final = test[selected_train_columns].to_numpy()
    final_predictions = np.floor(np.expm1(xgb.predict(X_final)))
    submission = test[['PropertyID']]
    submission['Predictions'] = final_predictions
    submission.to_csv('submission.csv', index=False)



############################################################################
########## Not really successful attempts with lasso and ridge #############
############################################################################

    # imputing values since here missing are not handled
    train.fillna(train.median(), inplace=True)
    X = train[selected_train_columns].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    kfolds = KFold(n_splits=10, shuffle=True, random_state=27)
    alphas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    lasso = make_pipeline(StandardScaler(), LassoCV(max_iter=1e7, alphas=alphas, random_state=27, cv=kfolds))
    #lasso = Lasso(0.000001)
    lasso.fit(X_train, y_train)

    R2_lasso = r2_score(y_test, lasso.predict(X_test))
    mse_lasso = mean_squared_error(y_test, lasso.predict(X_test))
    print('Lasso')
    print(R2_lasso)
    print(mse_lasso)
    print()

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=kfolds))
    #ridge = Ridge(0.000001)
    ridge.fit(X_train, y_train)

    R2_ridge = r2_score(y_test, ridge.predict(X_test))
    mse_ridge = mean_squared_error(y_test, ridge.predict(X_test))
    print('Ridge')
    print(R2_ridge)
    print(mse_ridge)
    print()


    




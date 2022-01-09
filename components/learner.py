from .data_modeler import DataModeler
from .data_preprocessor import DataPreprocessor

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
import lightgbm as lgb
import xgboost as xgb
import catboost as cat
import pandas as pd
import numpy as np
from typing import List
from tqdm import tqdm
import gc

class Learner(DataModeler):
    def __init__(self):
        super(Learner, self).__init__()

        self.model_dict = {
            'LR':    {'class':LinearRegression,      'model':[]},
            'Ridge': {'class':Ridge,                 'model':[]},
            'Lasso': {'class':Lasso,                 'model':[]},
            'DT':    {'class':DecisionTreeRegressor, 'model':[]},
            'KNN':   {'class':KNeighborsRegressor,   'model':[]},
            'RF':    {'class':RandomForestRegressor, 'model':[]},
            'SVM':   {'class':SVR,                   'model':[]},
            'LGB':   {'class':None,                  'model':[]},
            'XGB':   {'class':None,                  'model':[]},
            'CAT':   {'class':None,                  'model':[]}
        }
    
    def _lgb_train(self, train_x:pd.DataFrame, train_y:pd.DataFrame, valid_x:pd.DataFrame, valid_y:pd.DataFrame, params:dict, cat_feats:List[str]):
        """
        Train LGB models
        @param train_x: training data
        @param train_y: training label
        @param valid_x: validating data
        @param valid_y: validating label
        @param params: LGB model parameters
        """
        trn_data = lgb.Dataset(train_x, label=train_y)
        val_data = lgb.Dataset(valid_x, label=valid_y)
        model = lgb.train(
            params, 
            trn_data, 
            5000, 
            valid_sets = [trn_data, val_data],
            early_stopping_rounds = 20,
            categorical_feature=cat_feats
        )
        valid_pred = model.predict(valid_x, num_iteration=model.best_iteration)

        return model, valid_pred

    def _xgb_train(self, train_x:pd.DataFrame, train_y:pd.DataFrame, valid_x:pd.DataFrame, valid_y:pd.DataFrame, params:dict):
        """
        Train XGB models
        @param train_x: training data
        @param train_y: training label
        @param valid_x: validating data
        @param valid_y: validating label
        @param params: LGB model parameters
        """
        trn_data = xgb.DMatrix(train_x, train_y)
        val_data = xgb.DMatrix(valid_x, valid_y)

        watchlist = [(trn_data, 'train'), (val_data, 'valid')]
        model = xgb.train(dtrain=trn_data, num_boost_round=10000, 
                        evals=watchlist, early_stopping_rounds=20,
                        verbose_eval=500, params=params)

    
        valid_pred = model.predict(xgb.DMatrix(valid_x), ntree_limit=model.best_ntree_limit)

        return model, valid_pred

    def _cat_train(self, train_x:pd.DataFrame, train_y:pd.DataFrame, valid_x:pd.DataFrame, valid_y:pd.DataFrame, params:dict, cat_feats:List[str]):
        """
        Train CatBoost models
        @param train_x: training data
        @param train_y: training label
        @param valid_x: validating data
        @param valid_y: validating label
        @param params: LGB model parameters
        """
        cat_feat_indx = []
        train_x[cat_feats] = train_x[cat_feats].astype('str')
        valid_x[cat_feats] = valid_x[cat_feats].astype('str')
        for i,c in enumerate(train_x.columns):
            if c in cat_feats:
                cat_feat_indx.append(i)
        model = cat.CatBoostRegressor(iterations=20000, **params)
        model.fit(
            train_x.values, train_y.values, 
            eval_set=(valid_x.values, valid_y.values),
            cat_features=cat_feat_indx,
            use_best_model=True, 
            verbose=500
        )

        valid_pred = model.predict(valid_x.values)

        return model, valid_pred


    def train(self, df:pd.DataFrame, target:str, num_feats:List[str], cat_feats:List[str], nfold:int, model_name:str, model_param:dict, metric_func):
        """
        Train models
        @param df: trainset
        @param target: predicting target column name
        @param nfold: cross validation fold number
        @param num_feats: numerical features used for training mdoel
        @param cat_feats: categorical features used for training model (*only supported with LGB/Cat model)
        @param model_name: model name, must be one of keys in model_dict
        @param model_param: model parameter dict
        @param metric_func: metric function
        """
        assert model_name in self.model_dict.keys(), f"{model_name} is not supported"
        
        metric_l = []
        self.model_dict[model_name]['model'] = []

        df = DataPreprocessor.create_folds(df, nfold)
        df['pred'] = -1
        for fold in tqdm(range(nfold)):
            train_x = df[num_feats+cat_feats][df['fold']!=fold]
            train_y = df[target][df['fold']!=fold]
            valid_x = df[num_feats+cat_feats][df['fold']==fold]
            valid_y = df[target][df['fold']==fold]

            if model_name not in ['LGB', 'XGB', 'CAT']:
                model = self.model_dict[model_name]['class'](**model_param).fit(train_x[num_feats].fillna(0), train_y)
                valid_pred = model.predict(valid_x[num_feats].fillna(0))
            elif model_name == 'LGB':
                model, valid_pred = self._lgb_train(train_x, train_y, valid_x, valid_y, model_param, cat_feats)
            elif model_name == 'XGB':
                model, valid_pred = self._xgb_train(train_x[num_feats], train_y, valid_x[num_feats], valid_y, model_param)
            elif model_name == 'CAT':
                model, valid_pred = self._cat_train(train_x, train_y, valid_x, valid_y, model_param, cat_feats)
            
            df['pred'][df['fold']==fold] = valid_pred
            metric = metric_func(valid_y, valid_pred)
            metric_l.append(metric)
            self.model_dict[model_name]['model'].append(model)
            gc.collect()
        
        print(f"{model_name} End of training, avg metric: {np.mean(metric_l)}")
        return df.pop('pred')
    
    def _single_predict(self, df:pd.DataFrame, num_feats:List[str], cat_feats:List[str], model_name:str, is_train=False) -> np.array:
        """
        Use single model to predict
        @param df: testset
        @param num_feats: numerical features used for training mdoel
        @param cat_feats: categorical features used for training model (*only supported with LGB model)
        @param model_name: model name, must be one of keys in model_dict and has been trained
        @param is_train: if the input data is trainset
        return: prediction
        """
        pred_l = []
        if is_train:
            df = DataPreprocessor.create_folds(df, len(self.model_dict[model_name]['model']))
            df[model_name] = 0
        else:
            df['fold'] = -1
        for i,model in enumerate(self.model_dict[model_name]['model']):
            mask = (df['fold'] == -1) | (df['fold'] == i)
            if model_name not in ['LGB', 'XGB', 'CAT']:
                pred = model.predict(df[num_feats][mask].fillna(0)).tolist()
            elif model_name == 'LGB':
                pred = model.predict(df[num_feats+cat_feats][mask], num_iteration=model.best_iteration).tolist()
            elif model_name == 'XGB':
                pred = model.predict(xgb.DMatrix(df[num_feats][mask]), ntree_limit=model.best_ntree_limit).tolist()
            elif model_name == 'CAT':
                df_ = df.copy()
                df_[cat_feats] = df_[cat_feats].astype('str')
                pred = model.predict(df_[num_feats+cat_feats][mask].values).tolist()
            if is_train:
                df[model_name][df['fold']==i] = pred
            else:
                pred_l.append(pred)
        if is_train:
            pred = df.pop(model_name).values
        else:
            pred = np.mean(np.array(pred_l), axis=0)
        return pred
    
    def predict(self, df:pd.DataFrame, num_feats:List[str], cat_feats:List[str], models:List[str]=None, weights:List[float]=None, is_train=False) -> np.array:
        """
        Use all the models that have been trained before to predict
        @param df: testset
        @param num_feats: numerical features used for training mdoel
        @param cat_feats: categorical features used for training model (*only supported with LGB model)
        @param models: dict of model names to predict {model_name:weight}, if None then use all models
        @param weights: weights of models when embedding results of different models
        @param is_train: if the input data is trainset
        return: prediction
        """
        if models is not None:
            for m in models:
                assert m in self.model_dict.keys() and len(self.model_dict[m]['model'])>0, f"{m} is not trained."

        if models is None:
            #* if models are not assigned then use all trained models
            models = [x for x in self.model_dict.keys() if len(self.model_dict[x]['model'])>0]
        if weights is None:
            #* if weight is not assigned then take avg as default
            weights = [1/len(models) for _ in models]

        pred_l = []
        for i,model in tqdm(enumerate(models), "Predicting"):
            tmp = self._single_predict(df, num_feats, cat_feats, model, is_train) * weights[i]
            pred_l.append(tmp.tolist())
        
        pred = np.sum(np.array(pred_l), axis=0)
        return pred
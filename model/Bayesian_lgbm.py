# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import gc
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from hyperopt import fmin,tpe,hp,partial,space_eval
from sklearn.model_selection import KFold, StratifiedKFold
from hyperopt import STATUS_OK
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
from sklearn.metrics import average_precision_score


train_df = pd.read_csv('creditcard_train.csv')


#特征工程1:时间特征处理
timedelta = pd.to_timedelta(train_df['Time'], unit='s')
train_df['Minute'] = (timedelta.dt.components.minutes).astype(int)
train_df['Hour'] = (timedelta.dt.components.hours).astype(int)


feats = [f for f in train_df.columns if f not in ['Class','Index']]
folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

def average_precision_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    score_vali = average_precision_score(labels, preds)
    return 'average_precision_score', score_vali, True


def kfold_lightgbm(params):

    oof_preds = np.zeros(train_df.shape[0])

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Class'])):

        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df['Class'].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df['Class'].iloc[valid_idx],
                             free_raw_data=False, silent=True)


        clf = lgb.train(
            params=params,
            train_set=dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dvalid],
            early_stopping_rounds=20,
            feval=average_precision_score_vali,
            verbose_eval=False
        )

        oof_preds[valid_idx] = clf.predict(dvalid.data)

        del clf, dtrain, dvalid
        gc.collect()

    return average_precision_score(train_df['Class'], oof_preds)


#objective function
def lgb_objective(params,n_folds=5):

    loss = -kfold_lightgbm(params)

    return {'loss':-loss,'params':params,'status':STATUS_OK}

# Define the search space
space = {
    'objective':'regression',
    'boosting_type': 'gbdt',
    'subsample':0.8,
    'colsample_bytree':hp.uniform('colsample_bytree',0.8,0.9),
    'max_depth':7,
    'learning_rate':0.01,
    "lambda_l1":hp.uniform('lambda_l1',0.0,0.2),
    'seed':0,
}

#define algorithm
tpe_algorithm = tpe.suggest

best = fmin(fn = lgb_objective,space = space,algo=tpe_algorithm,max_evals=50)

print(best)
result=space_eval(space, best)
print(space_eval(space, best))
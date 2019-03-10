# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold
import gc
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns
import warnings
from imblearn.over_sampling import SMOTE
from collections import Counter
import time
from contextlib import contextmanager
warnings.simplefilter(action='ignore', category=FutureWarning)

train = pd.read_csv('creditcard_train.csv')
test = pd.read_csv('creditcard_test.csv')

df = pd.concat([train,test],axis=0)

#特征工程1:时间特征处理
timedelta = pd.to_timedelta(df['Time'], unit='s')
df['Minute'] = (timedelta.dt.components.minutes).astype(int)
df['Hour'] = (timedelta.dt.components.hours).astype(int)


#特征工程2:groupby特征
add = pd.DataFrame(df.groupby(["Minute"])['V14'].agg(['mean','std'])).reset_index()
add.columns = ["Minute","Minute_V14_MEAN", "Minute_V14_STD"]
df = df.merge(add, on=["Minute"], how="left")

#特征工程3:特征提取
# df['V14_is>1']=(df['V14']>1).astype(int)
# df['V10_is>2']=(df['V10']>2).astype(int)
# df['V28_is>1']=(df['V28']>1).astype(int)
# df['V5_is>6']=(df['V5']>6).astype(int)
# df['V6_is>4']=(df['V12']>4).astype(int)
# df['V7_is>6']=(df['V7']>6).astype(int)
# df['V9_is>1.5']=(df['V9']>1.5).astype(int)

#PIMP算法选择的特征
#drop_features=['V6_is>4','V7_is>6','V20','V1','V19','V17','V16','V15','Time','V24','Minute_V14_STD','Minute_V14_MEAN','Hour','Minute','V25','V28','Amount']

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance",
                                                                                                   ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')


def display_PR_Curve(y_true,y_score,average_precision):

    fig = plt.figure(figsize=(12, 6))
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.step(recall, precision, color='r', alpha=0.2,where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='#f25269')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: \n Average Precision-Recall Score ={0:0.3f}'.format(average_precision),fontsize=16)
    plt.savefig('lgbm_PR_Curve_bayesi.png',dpi=300)


def average_precision_score_vali(preds, data_vali):
    labels = data_vali.get_label()
    score_vali = average_precision_score(labels, preds)
    return 'average_precision_score', score_vali, True


# print('Resampled dataset shape {}'.format(Counter(y)))
def kfold_lightgbm(df):
    # Divide in training/validation and test data
    train_df = df[df['Class'].notnull()]
    test_df = df[df['Class'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    folds = StratifiedKFold(n_splits=5,shuffle=True,random_state=0)
    # Create arrays and dataframes to store results
    cv_result = np.zeros(5)
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])

    # def f(y):
    #     sample_dict = {}
    #     sample_dict[1] = 10000
    #     return sample_dict
    #
    # sm = SMOTE(ratio=f, random_state=42)

    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['Class','Index']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['Class'])):

        # X,y=sm.fit_sample(train_df[feats].iloc[train_idx],train_df['Class'].iloc[train_idx])

        dtrain = lgb.Dataset(data=train_df[feats].iloc[train_idx],
                             label=train_df['Class'].iloc[train_idx],
                             free_raw_data=False, silent=True)
        dvalid = lgb.Dataset(data=train_df[feats].iloc[valid_idx],
                             label=train_df['Class'].iloc[valid_idx],
                             free_raw_data=False, silent=True)


        params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'nthread': 4,
            'learning_rate': 0.01,
            'colsample_bytree':0.84, #0.8290813068817113 #贝叶斯优化参数
            'subsample': 0.9,
            'max_depth':7,
            'lambda_l1':0.15,#0.16277128856691855
            'seed':0,
            'verbose': -1,
        }

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
        sub_preds += clf.predict(test_df[feats]) / 5

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d average_precision_score : %.6f' % (n_fold + 1, average_precision_score(dvalid.label, oof_preds[valid_idx])))
        cv_result[n_fold]=average_precision_score(dvalid.label, oof_preds[valid_idx])
        del clf, dtrain, dvalid
        gc.collect()
    average_precision = average_precision_score(train_df['Class'], oof_preds)
    display_PR_Curve(train_df['Class'],oof_preds,average_precision)
    print('Full Local average_precision_score %.6f' % average_precision)
    print('CV std:{}'.format(cv_result.std()))

    display_importances(feature_importance_df)
    feature_importance_df = feature_importance_df.groupby('feature')['importance'].mean().reset_index().rename(index=str, columns={'importance': 'importance_mean'})
    feature_importance_df = feature_importance_df.sort_values(by='importance_mean', ascending=False)
    feature_importance_df.to_csv('feature_importance.csv')

    sub_df = pd.DataFrame({'Index': test['Index'].copy(),
                           'Pred': sub_preds})
    sub_df['Index'] = sub_df['Index'].astype(np.int64)
    sub_df['Class'] = (sub_df['Pred']>0.6).astype(np.int64)
    sub_df[['Index','Pred','Class']].to_csv('lgb_best.csv')

with timer("Process credit card balance"):
    kfold_lightgbm(df)
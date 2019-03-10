from sklearn.ensemble import IsolationForest
import numpy as np
import pandas  as pd

data = pd.read_csv('高端客户流失数据.csv',encoding='GBK')

#查看label类别分别
print(data['是否流失'].value_counts())

#分离特征与label
feats = [feat for feat in data.columns if feat not in ['是否流失']]
X = data[feats]
y = data['是否流失']

#模型训练 第二个参数可以权衡查准与召回
clf = IsolationForest(n_estimators=50,contamination=0.8,random_state=0)
clf.fit(data[feats])
train_pre = clf.predict(data[feats])

data['预测label']=train_pre

#流失 预测流失 个数
TP = data[(data['是否流失']==1)&(data['预测label']==-1)].shape[0]
#未流失 预测未流失
TN = data[(data['是否流失']==0)&(data['预测label']==1)].shape[0]
#流失 预测未流失
FN = data[(data['是否流失']==1)&(data['预测label']==1)].shape[0]
#未流失 预测流失
FP = data[(data['是否流失']==0)&(data['预测label']==-1)].shape[0]

#召回率 R=TP/TP+FN 查准率 P =TP/TP+FP

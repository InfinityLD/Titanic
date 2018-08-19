# coding: utf-8

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

label = train.Survived.values
train = train.drop(['Survived'], axis=1)


# --------------------- generate prediction column ------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2018).split(train, label)

cnt = 0
for train_index, test_index in skf:
    X_train = train.iloc[train_index]
    X_test = train.iloc[test_index]
    y_train = label[train_index]
    y_test = label[test_index]

    d_train = xgb.DMatrix(X_train, label=y_train)
    d_val = xgb.DMatrix(X_test, label=y_test)
    d_test = xgb.DMatrix(test)

    params={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric':'logloss',
    'gamma':0.1,
    'min_child_weight':1.1,
    'max_depth':5,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'colsample_bylevel':0.7,
    'eta': 0.01,
    'tree_method':'exact',
    'seed':0,
    'nthread':12,
    'silent': 1
    }

    model = xgb.train(params,d_train, num_boost_round=2000)
    
    test_pred = model.predict(d_test)
    test_prediction = pd.DataFrame(columns=['PassengerId'])
    test_prediction['PassengerId'] = test.PassengerId
    test_prediction['pred_'+str(cnt)] = test_pred
    test_prediction.to_csv('data/test_pred'+str(cnt)+'.csv', index=None) 
    
    pred = pd.DataFrame(columns=['PassengerId', 'prediction'])
    pred['PassengerId'] = X_test.PassengerId
    pred['prediction'] = model.predict(d_val)
    pred.to_csv('data/pred_' + str(cnt) + '.csv', index=None)
    
    print(cnt)
    cnt += 1

# -------------------------------- loading prediction files -------------------------------------

pred_0 = pd.read_csv('data/pred_0.csv')
pred_1 = pd.read_csv('data/pred_1.csv')
pred_2 = pd.read_csv('data/pred_2.csv')
pred_3 = pd.read_csv('data/pred_3.csv')
pred_4 = pd.read_csv('data/pred_4.csv')

test_pred_0 = pd.read_csv('data/test_pred0.csv')
test_pred_1 = pd.read_csv('data/test_pred1.csv')
test_pred_2 = pd.read_csv('data/test_pred2.csv')
test_pred_3 = pd.read_csv('data/test_pred3.csv')
test_pred_4 = pd.read_csv('data/test_pred4.csv')

test_pred = pd.merge(test_pred_0, test_pred_1, on='PassengerId', how='left')
test_pred = pd.merge(test_pred, test_pred_2, on='PassengerId', how='left')
test_pred = pd.merge(test_pred, test_pred_3, on='PassengerId', how='left')
test_pred = pd.merge(test_pred, test_pred_4, on='PassengerId', how='left')

avg_prediction= test_pred.drop('PassengerId', axis=1).mean(1)
test_pred['prediction'] = avg_prediction
test_pred = test_pred.drop(['pred_0', 'pred_1', 'pred_2', 'pred_3', 'pred_4'], axis=1)
test = pd.merge(test, test_pred, on='PassengerId', how='left')

pred = pd.concat([pred_0, pred_1, pred_2, pred_3, pred_4], axis=0, ignore_index=True)

train = pd.merge(train, pred, on='PassengerId', how='left')
print(train.head())
print(test.head())


# -------------------------- loading files with prediction column and train + predict ----------------------------

d_train = xgb.DMatrix(train, label=label)
d_test = xgb.DMatrix(test)

params={'booster':'gbtree',
'objective': 'binary:logistic',
'eval_metric':'logloss',
'gamma':0.1,
'min_child_weight':1.1,
'max_depth':5,
'lambda':10,
'subsample':0.7,
'colsample_bytree':0.7,
'colsample_bylevel':0.7,
'eta': 0.01,
'tree_method':'exact',
'seed':0,
'nthread':12,
'silent': 1
}

watchlist = [(d_train,'train')]
model = xgb.train(params,d_train, num_boost_round=1000, evals=watchlist)

test['label'] = model.predict(d_test)

def trans(pred):
    if pred >= 0.5:
        return 1
    else:
        return 0
    
test['Survived'] = test.label.apply(trans)


print(test.groupby('Survived').count())


submission = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission['PassengerId'] = test.PassengerId
submission['Survived'] = test.Survived
submission.to_csv('submission_0.2791.csv', index=None)
submission.describe()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)

df_train=pd.read_csv('df_train.csv')
df_test=pd.read_csv('df_test.csv')
print('read train and test data\n')

y=df_train['target'].values
X=df_train.drop(['msno','song_id','target'],axis=1)


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=1)
print('data has split')
print('X_train head()\n',X_train.head(2))
print('y_train head()\n',y_train)
print('X_val head()\n',X_val.head(2))
print('y_val head()\n',y_val)
#print('val head()\n',val.head())

#X_val.drop(['msno','song_id'],axis=1,inplace=True)

song_ids=df_test['id'].values
X_test=df_test.drop(['msno','song_id','id'],axis=1).values

lgb_train=lgb.Dataset(X_train,y_train)
lgb_val=lgb.Dataset(X_val,y_val,reference=lgb_train)

params={
        'boosting':'gbdt',
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.2,
        'num_leaves':128,
        'max_depth':10,
        'num_rounds':200,
        'begging_freq':1,
        'begging_seed':1,
        'max_bin':256,
        'n_jobs':-1
}
model_1=lgb.train(params=params,
                train_set=lgb_train,
                valid_sets=lgb_val,
                early_stopping_rounds=5)

params={
        'boosting':'dart',
        'objective':'binary',
        'metric':'auc',
        'learning_rate':0.2,
        'num_leaves':128,
        'max_depth':10,
        'num_rounds':200,
        'begging_freq':1,
        'begging_seed':1,
        'max_bin':256,
        'n_jobs':-1
}
model_2=lgb.train(params=params,train_set=lgb_train,valid_sets=lgb_val,early_stopping_rounds=5)

'''
#best param {'max_depth': 17, 'num_leaves': 58}
#best estimator LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.8,
#        importance_type='split', learning_rate=0.1, max_depth=17,
#       min_child_samples=20, min_child_weight=1, min_split_gain=0.0,
#      n_estimators=100, n_jobs=-1, num_leaves=58, objective='binary',
#      random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#      subsample=0.8, subsample_for_bin=200000, subsample_freq=0)'''

y_preds_1=model_1.predict(X_test,num_iteration=model_1.best_iteration)
y_preds_2=model_2.predict(X_test,num_iteration=model_2.best_iteration)
y_preds_avg=np.mean([y_preds_1,y_preds_2],axis=0)


#print(y_preds)
#print('auc is:',model.best_score_)

result_df=pd.DataFrame()
result_df['id']=song_ids
result_df['target']=y_preds_1

#保存结果
result_df.to_csv('submission.csv',index=False,
                float_format='%.5f')


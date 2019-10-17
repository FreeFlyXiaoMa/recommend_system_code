import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)

import codecs
import pandas as pd


train=pd.read_csv(r'D://recommend_system_code/data/train.csv',
                  names=['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship', 'race', 'gender',
                  'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                  'income_bracket'])

val=pd.read_csv(r'D://recommend_system_code/data/test.csv',names=[
                  'age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship', 'race', 'gender',
                  'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                  'income_bracket'])

test=pd.read_csv(r'D://recommend_system_code/data/census_input.csv',names=[
                  'age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship', 'race', 'gender',
                  'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
                  ])

# =====================================================================================================
# 特征工程
# =====================================================================================================

#将年龄分为一个范围,方便转换为类别型特征
train['age_range']=pd.cut(train['age'],bins=[15,30,45,60,75,90,105])
val['age_range']=pd.cut(val['age'],bins=[15,30,45,60,75,90,105])
test['age_range']=pd.cut(test['age'],bins=[15,30,45,60,75,90,105])

combine=[train,val,test]
for value in combine:
    value.loc[(value['age'] > 15) & (value['age'] <= 30), 'age_category'] = 0
    value.loc[(value['age'] > 30) & (value['age'] <= 45), 'age_category'] = 1
    value.loc[(value['age'] > 45) & (value['age'] <= 60), 'age_category'] = 2
    value.loc[(value['age'] > 60) & (value['age'] <= 75), 'age_category'] = 3
    value.loc[(value['age'] > 75) & (value['age'] <= 90), 'age_category'] = 4
    value.loc[(value['age'] > 90) & (value['age'] <= 105), 'age_category'] = 5

train.drop(['age','age_range'],axis=1,inplace=True)
val.drop(['age','age_range'],axis=1,inplace=True)
test.drop(['age','age_range'],axis=1,inplace=True)


def flat_workclass(x):
    if x=='?':
        return 'Other_workclass'
    elif x==None:
        return 'Other_workclass'
    else:return x
train['workclass']=train['workclass'].apply(lambda x:flat_workclass(x))
val['workclass']=val['workclass'].apply(lambda x:flat_workclass(x))
test['workclass']=test['workclass'].apply(lambda x:flat_workclass(x))

train['workclass']=train['workclass'].astype('category')
val['workclass']=val['workclass'].astype('category')
test['workclass']=test['workclass'].astype('category')

train['education']=train['education'].astype('category')
val['education']=val['education'].astype('category')
test['education']=test['education'].astype('category')


train['education_num']=train['education_num'].astype('int')
val['education_num']=val['education_num'].astype('int')
test['education_num']=test['education_num'].astype('int')

train['marital_status']=train['marital_status'].astype('category')
val['marital_status']=val['marital_status'].astype('category')
test['marital_status']=test['marital_status'].astype('category')


def flat_occupation(x):
    if x=="?":
        return 'Other_occupation'
    else:
        return x

train['occupation']=train['occupation'].apply(lambda x:flat_occupation(x))
val['occupation']=val['occupation'].apply(lambda x:flat_occupation(x))
test['occupation']=test['occupation'].apply(lambda x:flat_occupation(x))


train['occupation']=train['occupation'].astype('category')
val['occupation']=val['occupation'].astype('category')
test['occupation']=test['occupation'].astype('category')


train['relationship']=train['relationship'].astype('category')
val['relationship']=val['relationship'].astype('category')
test['relationship']=test['relationship'].astype('category')

train['race']=train['race'].astype('category')
val['race']=val['race'].astype('category')
test['race']=test['race'].astype('category')

train['gender']=train['gender'].astype('category')
val['gender']=val['gender'].astype('category')
test['gender']=test['gender'].astype('category')


train['capital_gain']=train['capital_gain'].astype('int')
val['capital_gain']=val['capital_gain'].astype('int')
test['capital_gain']=test['capital_gain'].astype('int')


train['capital_loss']=train['capital_loss'].astype('int')
val['capital_loss']=val['capital_loss'].astype('int')
test['capital_loss']=test['capital_loss'].astype('int')


train['hours_per_week']=train['hours_per_week'].astype('int')
val['hours_per_week']=val['hours_per_week'].astype('int')
test['hours_per_week']=test['hours_per_week'].astype('int')


train['native_country']=train['native_country'].astype('category')
val['native_country']=val['native_country'].astype('category')
test['native_country']=test['native_country'].astype('category')


def flat_target(x):
    """
    归一化标签值
    :param x:
    :return:
    """
    if x==">50K":
        return int(1)
    elif x=='<=50K':
        return int(0)
train['label']=train['income_bracket'].apply(lambda x:flat_target(x))
val['label']=val['income_bracket'].apply(lambda x:flat_target(x))


train.drop('income_bracket',axis=1,inplace=True)
val.drop('income_bracket',axis=1,inplace=True)

# =====================================================================================================
# 模型训练
# =====================================================================================================
y_train=train['label'].values
X_train=train.drop('label',axis=1)

print(test.head())












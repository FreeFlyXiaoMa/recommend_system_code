# -*- coding: utf-8 -*-
#@Time  :2019/9/20 上午11:33
#@Author  :XiaoMa

import numpy as np
from module.news.model_eda.LFM.read_ import get_train_data


def lfm_train(train_data,F,alpha,beta,step):
    """
    LFM model----user_vector  news_vector
    :param train_data:
    :param F:
    :param alpha:
    :param beta:
    :param step:
    :return:
    """
    user_vec={}
    item_vec={}

    for step_index in range(step):
        for data_instance in train_data:
            # print('data_instance:',data_instance)
            user_id,item_id,label=data_instance
            if user_id not in user_vec:
                user_vec[user_id]=init_model(F)
            if item_id not in item_vec:
                item_vec[item_id]=init_model(F)
            delta=label-model_predict(user_vec[user_id],item_vec[item_id])

            for index in range(F):
                user_vec[user_id][index]=user_vec[user_id][index]-beta*(-delta*item_vec[item_id][index]+alpha*user_vec[user_id][index])
                item_vec[item_id][index]=item_vec[item_id][index]-beta*(-delta*user_vec[user_id][index]+alpha*item_vec[item_id][index])

        beta=beta*0.9
    print('LFM----User Vector & Item Vector has trianed!!!')
    return user_vec,item_vec


def model_predict(user_vector,item_vector):
    """
    predict
    :param user_vec:
    :param item_vec:
    :return:
    """
    res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res

def init_model(vector_len):
    """

    :param vector_len:length of initial vector
    :return:
    """
    return np.random.randn(vector_len)  #Normal Distribution

def give_recom_result(user_vec,item_vec,user_id,fix_num):
    """
    Give fix user_id recom result
    :param user_vec:
    :param item_vec:
    :param user_id:
    :return:
    """

    if user_id not in user_vec: #This is a new user---user_cold_boot
        print('*'*20,'User_id:',user_id,'is a new user !!','*'*20)
        return []
    record={}
    recom_list=[]

    user_vector=user_vec[user_id]
    for itemid in item_vec:
        item_vector=item_vec[itemid]
        res=np.dot(user_vector,item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
        record[itemid]=res
    for zuhe in sorted(record.items(),key=lambda x:x[1],reverse=True)[:fix_num]:    #reorder the item_id by score
        item_id=zuhe[0]
        score=round(zuhe[1],3)
        recom_list.append((item_id,score))
    # print('recomm_list[0:5],',recom_list[:5])
    # print('len(recom_list):',len(recom_list))
    return recom_list


def model_train_process(user_id,recom_num,F,alpha,beta,step):
    """

    :return:
    """
    # print('user_id=====',user_id)
    train_data=get_train_data('./module/news/model_eda/LFM/data/ratings.csv')
    user_vec,item_vec=lfm_train(train_data,F,alpha,beta,step)
    recom_list=give_recom_result(user_vec,item_vec,user_id,recom_num)

    return recom_list

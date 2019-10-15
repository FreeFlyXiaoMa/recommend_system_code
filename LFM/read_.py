# -*- coding: utf-8 -*-
#@Time  :2019/9/20 下午12:06
#@Author  :XiaoMa
import os

def get_train_data(input_file):
    if not os.path.exists(input_file):
        print('Input_file is not exists:',input_file)
        return {}
    score_dict=get_ave_score(input_file)
    # print('score_dict:',score_dict)
    neg_dic={}
    pos_dic={}
    train_data=[]
    linenum=0
    with open(input_file,encoding='utf-8') as fp:
        for line in fp:
            if linenum ==0:
                linenum+=1
                continue
            item=line.strip().split(',')
            # print('item:',item)
            if len(item)<3:
                continue
            item_id,user_id,rating=item[1],item[2],float(item[3])
            if user_id not in pos_dic:
                pos_dic[user_id]=[]
            if user_id not in neg_dic:
                neg_dic[user_id]=[]

            if rating>=4.0:
                pos_dic[user_id].append((item_id,1.0))
            else:
                score=score_dict.get(item_id,0)
                neg_dic[user_id].append((item_id,score))
        # print('neg_dic:',neg_dic)
        # print('pos_dic:',pos_dic)
    for userid in pos_dic:
        data_num=min(len(pos_dic[userid]),len(neg_dic.get(userid,[])))
        if data_num>0:
            train_data+=[(userid,zuhe[0],zuhe[1]) for zuhe in pos_dic[userid]][:data_num]
        else:
            continue
        sorted_neg_list=sorted(neg_dic[userid],key=lambda element:element[1],reverse=True)
        for zuhe in sorted_neg_list:
            train_data+=[(userid,zuhe[0],0)]
    return train_data


def get_ave_score(input_file):
    if not os.path.exists(input_file):
        return {}
    linenum=0
    record_dict={}
    score_dict={}
    tp=open(input_file,encoding='utf-8')

    for line in tp:
        if linenum==0:
            linenum+=1
            continue
        item=line.strip().split(',')
        if len(item)<3:
            continue
        item_id,user_id,rating=item[1],item[2],float(item[3])
        # print('rating:',rating)
        if item_id not in record_dict:
            record_dict[item_id]=[0,0]
        record_dict[item_id][0]+=1
        record_dict[item_id][1]+=rating
    # print('record_dict:',record_dict)
    tp.close()

    for itemid in record_dict:
        score_dict[itemid]=round(record_dict[itemid][1]/record_dict[itemid][0],3)

    # print('score_dict:',score_dict)
    return score_dict


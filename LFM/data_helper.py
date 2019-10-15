# -*- coding: utf-8 -*-
#@Time  :2019/9/17 下午5:08
#@Author  :XiaoMa

import numpy as np
import pandas as pd
import pickle

import argparse
import pymysql
from sqlalchemy import create_engine

from module.news.model_eda.LFM.utils import flatten_to_target

pymysql.install_as_MySQLdb()

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',1000)
pd.set_option('display.max_colwidth',1000)
pd.set_option('display.max_rows',1000)

# parser = argparse.ArgumentParser()
# parser.add_argument('--sql_news', default=True, help='get the news corpus')
# parser.add_argument('--sql_user_log', default=True, help="get the user's action log")
# parser.add_argument('--news_content', default=False, help='store news content')
# parser.add_argument('--predict', default=True, help='predict news category')
# args = parser.parse_args()

def lfm_data_eda(sql_user_log=False):
    # sql_news=True

    print('Run the data_helper !!!')

    # #get news data
    # if sql_news:
    #     print('Connect the database!')
    #     engine=create_engine('mysql://hhm:123456@192.168.3.88:3306/hhmDB')
    #     conn=engine.connect()
    #     df_news=pd.read_sql_table(table_name='summary_news',con=conn)
    #
    #     pickle.dump(df_news,open('news.pkl','wb'))
    # else:
    #     df_news=pickle.load(open('./data/news.pkl','rb'))

    #------------------------------------------Logs--------------------------------------------
    print('***Load the User Log***')
    if sql_user_log:
        engine=create_engine('mysql://hhm:123456@222.128.117.35:3306/hhmDB')
        conn=engine.connect()

        table_log=pd.read_sql_table(table_name='UserAction',con=conn)
        # print(str(table_log['user_id']))
                # drop the 95 user logs
        for i in range(95):
            table_log.drop(i, inplace=True)

        pickle.dump(table_log,open('./module/news/model_eda/LFM/data/table_log.pkl','wb'))

        news=pd.read_sql_table(table_name='summary_news',con=conn)
        pickle.dump(news,'./data/summary_news.pkl')

    else:
        table_log=pickle.load(open('./module/news/model_eda/LFM/data/table_log.pkl','rb'))
    table_log.drop(['id'],axis=1,inplace=True)


    table_log['target']=table_log['read_times'].apply(lambda x:flatten_to_target(x))

    pickle.dump(table_log,open('./module/news/model_eda/LFM/data/table_log_target.pkl','wb'))
    #read_times    login_time     exit_time  give_thumbs equipment Browser        news_id        user_id target
    table_log.drop(['read_times','login_time','exit_time','give_thumbs','equipment','Browser'],axis=1,inplace=True)
    table_log.reset_index(inplace=True)
    table_log.drop(['index'],axis=1,inplace=True)

    table_log.to_csv(open('./module/news/model_eda/LFM/data/ratings.csv','w',encoding='utf-8'))


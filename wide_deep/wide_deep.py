# -*- coding: utf-8 -*-
# @Time     :2019/10/14 11:14
# @Author   :XiaoMa
# @File     :wide_deep.py

import tensorflow as tf
import sys
import time,os
import shutil
import argparse

#,id,read_times,login_time,exit_time,give_thumbs,equipment,
# Browser,target,news_type,news_source,news_author,news_redactor,
# news_keywords,img_count,thumbs_number,read_count,pubdate_year,pubdate_month,
# pubdate_day,pubdate_hour,pubdate_minute
#21
parser=argparse.ArgumentParser()
parser.add_argument('--model_type',type=str,default='wide_deep',help="模型类型：{'wide','deep','wide_deep'}")
parser.add_argument('--train_epoch',type=int,default=3,help='训练的迭代次数')
parser.add_argument('--epoch_per_eval',type=int,default=2,help='训练的迭代步数')
parser.add_argument('--batch_size',type=int,default=64,help='没批次的样本数')
parser.add_argument('--train_data',type=str,default='train_data.csv')
parser.add_argument('--test_data',type=str,default='test_data.csv')
args=parser.parse_args()

_CSV_COLUMNS_NAME=['read_times',
                   # 'login_time','exit_time',
                   'give_thumbs',#'equipment','Browser',
                   'target',#'news_type','news_source','news_author','news_redactor','news_keywords','img_count',
                   #'thumbs_number','read_count','pubdate_year','pubdate_month','pubdate_day','pubdate_hour',
                   'pubdate_minute'
                   ]

_CSV_COLUMN_DEFAULTS=[[0.0],
                      # [0.0],[0.0],
                      [0],#[''],[''],
                      [0],#[''],[''],[''],[''],[''],[0],
                      #[0],[0],[0],[0],[0],[0],
                      [0],
                      ]

_NUM_EXAMPLES={
    'train':3
}

#tf的配置信息
# def get_session():
#     cfg=tf.ConfigProto(log_device_placement=False)  #获取到operations 和Tensor被指派到哪个设备
#     cfg.gpu_options.allow_growth=True  #程序用多少就占多少内存
#     return tf.Session(config=cfg)
#
# sess=get_session()

def build_feature_column():
    #wide and deepcolumn features

    #数值型特征有：read_times，login_time，exit_time，img_count，thumbs_number，read_count,pubdate_month,pubdate_day,
    # pubdate_hour,pubdate_minute

    read_times=tf.feature_column.numeric_column('read_times')
    # login_time=tf.feature_column.numeric_column('login_time')
    # exit_time=tf.feature_column.numeric_column('exit_time')
    # img_count=tf.feature_column.numeric_column('img_count')
    # thumbs_number=tf.feature_column.numeric_column('thumbs_number')
    # read_count=tf.feature_column.numeric_column('read_count')
    # month=tf.feature_column.numeric_column('pubdate_month')
    # day=tf.feature_column.numeric_column('pubdate_day')
    # hour=tf.feature_column.numeric_column('pubdate_hour')
    minute=tf.feature_column.numeric_column('pubdate_minute')

    #连续特征离散化
    # month_bucket=tf.feature_column.bucketized_column(month,boundaries=[3,6,9,12])
    # day_bucket=tf.feature_column.bucketized_column(day,boundaries=[10,20,30])
    # hour_bucket=tf.feature_column.bucketized_column(hour,boundaries=[6,12,18,24])
    minute_bucket=tf.feature_column.bucketized_column(minute,boundaries=[10,20,30,40,50,60])

    #类别型特征--vocabulary_list: give_thumbs,equipment,Browser,pubdate_year,
    give_thumbs=tf.feature_column.categorical_column_with_vocabulary_list(key='give_thumbs',vocabulary_list=[0,1],dtype=tf.int64)
    # equipment=tf.feature_column.categorical_column_with_vocabulary_list('equipment',vocabulary_list=['linux', 'Win', 'iPhone', 'mac'])
    # Browser=tf.feature_column.categorical_column_with_vocabulary_list('Browser',vocabulary_list=['Chrome', 'Safari'])
    # pubdate_year=tf.feature_column.categorical_column_with_vocabulary_list('pubdate_year',vocabulary_list=[2019,2020])

    #类别型特征--hash_bucket:news_type,news_source,news_author,news_redactor,,
    # news_type=tf.feature_column.categorical_column_with_hash_bucket(key='news_type',hash_bucket_size=500)
    # news_source=tf.feature_column.categorical_column_with_hash_bucket('news_source',hash_bucket_size=1000)
    # news_author=tf.feature_column.categorical_column_with_hash_bucket('news_author',hash_bucket_size=1000)
    # news_redactor=tf.feature_column.categorical_column_with_hash_bucket('news_redactor',hash_bucket_size=1000)
    # news_keywords=tf.feature_column.categorical_column_with_hash_bucket('news_keywords',hash_bucket_size=1000)

    # class_columns=[month_bucket,day_bucket,hour_bucket,minute_bucket,give_thumbs,equipment,Browser,pubdate_year,news_type,
    #                news_source,news_author,news_redactor,news_keywords]

    class_columns=[minute_bucket,give_thumbs]

    # crossed_columns=[tf.feature_column.crossed_column(['give_thumbs','news_type'],hash_bucket_size=500),
    #                  tf.feature_column.crossed_column(['give_thumbs','news_source'],hash_bucket_size=1000),
    #                  tf.feature_column.crossed_column(['news_type','news_source','news_author'],hash_bucket_size=1000)
    #                  ]
    #
    # wide_columns=class_columns+crossed_columns

    wide_columns=class_columns

    #稀疏向量转换为稠密向量
    # deep_columns=[read_times,login_time,exit_time,img_count,thumbs_number,read_count,
    #               tf.feature_column.indicator_column(give_thumbs),
    #               tf.feature_column.indicator_column(equipment),
    #               tf.feature_column.indicator_column(Browser),
    #               tf.feature_column.indicator_column(pubdate_year),
    #               tf.feature_column.indicator_column(news_type),
    #               tf.feature_column.indicator_column(news_source),
    #               tf.feature_column.indicator_column(news_author),
    #               tf.feature_column.indicator_column(news_redactor),
    #               tf.feature_column.embedding_column(news_keywords,dimension=20)
    # ]
    deep_columns=[read_times]

    return wide_columns,deep_columns

def build_estimator(model_dir,model_type):
    """
    为指定的模型构建estimator
    :param model_dir:
    :param model_type:
    :param run_config:
    :return:
    """
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    #生成feature column
    wide_columns,deep_columns=build_feature_column()
    hidden_units=[100,50]

    if model_type=='wide':
        model=tf.estimator.LinearClassifier(model_dir=model_dir,feature_columns=wide_columns,config=run_config)
    elif model_type=='deep':
        model=tf.estimator.DNNClassifier(model_dir=model_dir,feature_columns=deep_columns,config=run_config)
    else:
        model=tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir,
                                                       linear_feature_columns=wide_columns,
                                                       dnn_feature_columns=deep_columns,
                                                       dnn_hidden_units=hidden_units,
                                                       config=run_config)

    return model

def input_fn(data_file,num_epochs,shuffle,batch_size):
    """
    input function for the Estimator
    :param dataset:
    :param num_epochs:
    :param shuffle:
    :param batch_size:
    :return:
    """
    def parse_csv(value):

        columns=tf.decode_csv(value,record_defaults=_CSV_COLUMN_DEFAULTS)
        features=dict(zip(_CSV_COLUMNS_NAME,columns))
        labels=features.pop('target')

        return features,labels

    dataset=tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset=dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

    dataset=dataset.map(parse_csv,num_parallel_calls=5)

    #call repeat after shuffering,rather than before,to prevent separate epochs from blending together
    dataset=dataset.repeat(num_epochs)
    dataset=dataset.batch(batch_size)

    iterator=dataset.make_one_shot_iterator()
    features,labels=iterator.get_next()
    return features,labels

def tf_read_file(file):
    assert tf.gfile.Exists(file),print('{} is not found'.format(file))
    dataset=tf.data.TextLineDataset(file)   #每一个元素对应一行
    return dataset

def main():

    cur_model_dir="{}_{}_{}_{}_{}".format(args.model_type,args.batch_size,args.train_epoch,
                                          args.epoch_per_eval,str(int(time.time())))

    shutil.rmtree(cur_model_dir,ignore_errors=True)

    cfg=tf.ConfigProto(log_device_placement=False)
    cfg.gpu_options.allow_growth=True

    # run_cfg=tf.estimator.RunConfig().replace(session_config=cfg,
    #                                          keep_checkpoint_max=1,
    #                                          save_summary_steps=10000,
    #                                          save_checkpoints_steps=10000,
    #                                          log_step_count_steps=10000)

    model=build_estimator(cur_model_dir,args.model_type)

    train=tf_read_file(args.train_data)

    #训练
    for i in range(args.train_epoch//args.epoch_per_eval):
        start_time=time.clock()
        print('-'*60)
        print('#eval:',str(i+1))
        model.train(input_fn=lambda :input_fn(args.train_data,args.train_epoch,True,args.batch_size))
        # model.train(input_fn=train)
        end_time=time.clock()
        print('平均每个epoch花费时间：{}'.format(int(end_time-start_time)/args.epoch_per_eval))

        print('*'*20,'开始进行评估：','*'*20)
        results=model.evaluate(input_fn=lambda :input_fn(args.test_data,1,False,args.batch_size))
        print("# epoch_{} result:".format((i+1)*args.epoch_per_eval))
        #
        for key in sorted(results):
            print("%s : %s"%(key,results[key]))
    # 开始保存模型，为后续提供server服务(需要定义导出目录，用于模型的接收参数)
    wide_columns, deep_columns = build_feature_column()
    print("模型的输入列名：", wide_columns, deep_columns)
    feature_columns=wide_columns+deep_columns
    features_spec = tf.feature_column.make_parse_example_spec(feature_columns=feature_columns)
    print("从输入列名开始创建字典！ ")

    # 构建接收函数，并导出模型
    export_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(features_spec)
    print("已构建接收参数")
    servable_model_dir='./tmp'
    servable_model_path=model.export_savedmodel(servable_model_dir, export_input_fn)
    print("************ Done Exporting at Path -%s",servable_model_path)
if __name__=='__main__':
    # os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ['CUDA_VISIBLE_DEVICES']='0'
    main()




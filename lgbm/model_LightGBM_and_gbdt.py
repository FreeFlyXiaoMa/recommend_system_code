import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

df_songs_extra=pd.read_csv('song_extra_info.csv')
df_members=pd.read_csv('members.csv')
df_songs=pd.read_csv('songs.csv')

#将歌曲合并到训练集和测试集，合并标准是歌曲id
df_train=df_train.merge(df_songs,on='song_id',how='left')
df_test=df_test.merge(df_songs,on='song_id',how='left')

#合并用户信息
df_train=df_train.merge(df_members,on='msno',how='left')
df_test=df_test.merge(df_members,on='msno',how='left')
#---------------------------------特征工程部分---------------------------
#---------------------------------空值填充-------------------------------
df_train['gender'].fillna(value='Unknown',inplace=True)
df_test['gender'].fillna(value='Unknown',inplace=True)

df_train['source_screen_name'].fillna(value='Unknown',inplace=True)
df_test['source_screen_name'].fillna(value='Unknown',inplace=True)

df_train['source_type'].fillna(value='Unknown',inplace=True)
df_test['source_type'].fillna(value='Unknown',inplace=True)

df_train['genre_ids'].fillna(value='Unknown',inplace=True)
df_test['genre_ids'].fillna(value='Unknown',inplace=True)

df_train['composer'].fillna(value='Unknown',inplace=True)
df_test['composer'].fillna(value='Unknown',inplace=True)

df_train['lyricist'].fillna(value='Unknown',inplace=True)
df_test['lyricist'].fillna(value='Unknown',inplace=True)

#填充歌曲长度的均值
df_train['song_length'].fillna(value=df_train['song_length'].mean(),inplace=True)
df_test['song_length'].fillna(value=df_test['song_length'].mean(),inplace=True)

#语言类型填充出现频率最高的一种
df_train['language'].fillna(value=df_train['language'].mode()[0],inplace=True)
df_test['language'].fillna(value=df_test['language'].mode()[0],inplace=True)

#----------------------------------------空值填充后的处理-------------------------
#genre_ids
df_train['genre_ids']=df_train['genre_ids'].str.split('|')
df_test['genre_ids']=df_test['genre_ids'].str.split('|')
df_train['genre_count']=df_train['genre_ids'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
df_test['genre_count']=df_test['genre_ids'].apply(lambda x:len(x) if 'Unknown' not in x else 0)

#Artist
print('训练集中歌手数量：',df_train['artist_name'].unique().shape[0])
print('测试集中歌手数量：',df_test['artist_name'].unique().shape[0])
print('训练集和测试集中都出现的歌手数量有：',len(set.intersection(set(df_train['artist_name']),set(df_test['artist_name']))))
df_artists=df_train.loc[:,['artist_name','target']]
artist1=df_artists.groupby(['artist_name'],as_index=False).sum().rename(
    columns={'target':'repeat_count'}
)
artist2=df_artists.groupby(['artist_name'],as_index=False).count().rename(
    columns={'target':'play_count'}
)
#计算歌手出现的比例
df_artist_repeats=artist1.merge(artist2,how='inner',on='artist_name')
print(df_artist_repeats.head())
df_artist_repeats['repeat_percentage']=round(
    (df_artist_repeats['repeat_count']*100)/df_artist_repeats['play_count'],1)
print(df_artist_repeats.head())
df_artist_repeats.drop(['repeat_count','play_count'],axis=1,inplace=True)

#合并到训练集和测试集
df_train=df_train.merge(df_artist_repeats,on='artist_name',how='left').rename(
    columns={'repeat_percentage':'artist_repeat_percentage'})
df_test=df_test.merge(df_artist_repeats,on='artist_name',how='left').rename(
    columns={'repeat_percentage':'artist_repeat_percentage'}
)
df_test['artist_repeat_percentage'].fillna(value=0.0,inplace=True)


#特征处理后，去掉genre_ids，artist_name
df_train.drop(['genre_ids','artist_name'],axis=1,inplace=True)
df_test.drop(['genre_ids','artist_name'],axis=1,inplace=True)
del df_artist_repeats
del df_artists


#composer
df_train['composer']=df_train['composer'].str.split('|')
df_test['composer']=df_test['composer'].str.split('|')

df_train['composer']=df_train['composer'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
df_test['composer']=df_test['composer'].apply(lambda x:len(x) if 'Unknown' not in x else 0)

#source_system_tab
#查看source_system_tab有多少种类型
print('source_system_tab:\n',df_train['source_system_tab'].value_counts())

#采用映射的方式，类似于使用label-encoder编码
#思考：类别之间的重要性可能会不一样，所以采用类似于label-encoder
source_tab_dict={'my library':8,
                 'discover':7,
                 'search':6,
                 'radio':5,
                 'listen with':4,
                 'explore':3,
                 'notification':2,
                 'setting':1,
                 'Unknown':0
                 }

source_screen_name_dict={'Local playlist more':19,
                         'Online playlist more':18,
                         'Rdio':17,
                         'Unknown':16,
                         'Album more':15,
                         'Search':14,
                         'Artist more':13,
                         'Discover Feature':12,
                         'Discover Chart':11,
                         'Others profile more':10,
                         'Discover Genre':9,
                         'My library':8,
                         'Explore':7,
                         'Discover New':6,
                         'Search Trends':5,
                         'Search Home':4,
                         'My library_Search':3,
                         'Self profile more':2,
                         'Concert':1,
                         'Payment':0
}

source_type_dict={'local-library':12,
                  'online-playlist':11,
                  'local-playlist':10,
                  'radio':9,
                  'album':8,
                  'top-hits-for-artist':7,
                  'song':6,
                  'song-based-playlist':5,
                  'listen-with':4,
                  'Unknown':3,
                  'topic-article-playlist':2,
                  'artist':1,
                  'my-daily-playlist':0
}
df_train['source_system_tab']=df_train['source_system_tab'].map(source_tab_dict)
df_test['source_system_tab']=df_test['source_system_tab'].map(source_tab_dict)

df_train['source_screen_name']=df_train['source_screen_name'].map(source_screen_name_dict)
df_test['source_screen_name']=df_test['source_screen_name'].map(source_screen_name_dict)
df_test['source_screen_name'].fillna(df_test['source_screen_name'].mode()[0],inplace=True)

df_train['source_type']=df_train['source_type'].map(source_type_dict)
df_test['source_type']=df_test['source_type'].map(source_type_dict)

#one-hot编码方式
#思考：类别的重要性相同，采用类似one-hot编码方式
gender_train=pd.get_dummies(df_train['gender'],drop_first=True)
gender_test=pd.get_dummies(df_test['gender'],drop_first=True)

#拼接
df_train=pd.concat([df_train,gender_train],axis=1)
df_test=pd.concat([df_test,gender_test],axis=1)

#特征处理后，去掉无用的特征
df_train.drop(['composer','gender'],axis=1,inplace=True)
df_test.drop(['composer','gender'],axis=1,inplace=True)

#有效的时间
df_train['validaty_days']=pd.to_timedelta(df_train['expiration_date']-df_train['registration_init_time'],unit='d').dt.days
df_test['validaty_days']=pd.to_timedelta(df_test['expiration_date']-df_test['registration_init_time'],unit='d').dt.days
#df_train['validaty_days']=(df_train['expiration_date']-df_train['registration_init_time']).dt.days
#df_test['validaty_days']=(df_test['expiration_date']-df_test['registration_init_time']).dt.days
#处理后，去掉时间特征
df_train.drop(['expiration_date','registration_init_time'],axis=1,inplace=True)
df_test.drop(['expiration_date','registration_init_time'],axis=1,inplace=True)

#lyricist作词
df_train['lyricist']=df_train['lyricist'].str.split('|')
df_test['lyricist']=df_test['lyricist'].str.split('|')
df_train['lyricist_count']=df_train['lyricist'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
df_test['lyricist_count']=df_test['lyricist'].apply(lambda x:len(x) if 'Unknown' not in x else 0)
#处理后，去掉lyricist列
df_train.drop('lyricist',axis=1,inplace=True)
df_test.drop('lyricist',axis=1,inplace=True)


#歌曲额外信息
df_songs_extra.drop('name',axis=1,inplace=True)
#合并到训练集合测试集
df_train=df_train.merge(df_songs_extra,on='song_id',how='left')
df_test=df_test.merge(df_songs_extra,on='song_id',how='left')

#歌曲发行音像制品
def isrc_to_year(isrc):
    if type(isrc)==str:
        if int(isrc[5:7])>17:
            return 1900+int(isrc[5:7])
        else:
            return 2000+int(isrc[5:7])
    else:
        return 1950  #取1900、2000的中间值

df_train['song_year']=df_train['isrc'].apply(isrc_to_year)
df_test['song_year']=df_test['isrc'].apply(isrc_to_year)
#处理后去掉原特征
df_train.drop('isrc',axis=1,inplace=True)
df_test.drop('isrc',axis=1,inplace=True)
#将歌曲年份特征转换为数值型
df_train['song_year']=df_train['song_year'].astype('int')
df_test['song_year']=df_test['song_year'].astype('int')

#
df_train['source_system_tab']=df_train['source_system_tab'].astype('category')
df_test['source_system_tab']=df_test['source_system_tab'].astype('category')

df_train['source_screen_name']=df_train['source_screen_name'].astype('category')
df_test['source_screen_name']=df_test['source_screen_name'].astype('category')

df_train['source_type']=df_train['source_type'].astype('category')
df_test['source_type']=df_test['source_type'].astype('category')

df_train['language']=df_train['language'].astype('category')
df_test['language']=df_test['language'].astype('category')

df_train['city']=df_train['city'].astype('category')
df_test['city']=df_test['city'].astype('category')

df_train['registered_via']=df_train['registered_via'].astype('category')
df_test['registered_via']=df_test['registered_via'].astype('category')

#将年龄分为一个范围,方便转换为类别型特征
df_train['age_range']=pd.cut(df_train['bd'],bins=[-45,0,10,18,35,50,80,200])
df_test['age_range']=pd.cut(df_test['bd'],bins=[-45,0,10,18,35,50,80,200])

combine=[df_train,df_test]
for value in combine:
    value.loc[(value['bd'] > 0) & (value['bd'] <= 10), 'age_category'] = 0
    value.loc[(value['bd'] > 80) & (value['bd'] <= 200), 'age_category'] = 1
    value.loc[(value['bd'] > 50) & (value['bd'] <= 80), 'age_category'] = 2
    value.loc[(value['bd'] > 10) & (value['bd'] <= 18), 'age_category'] = 3
    value.loc[(value['bd'] > 35) & (value['bd'] <= 50), 'age_category'] = 4
    value.loc[(value['bd'] > -45) & (value['bd'] <= 0), 'age_category'] = 5
    value.loc[(value['bd'] > 18) & (value['bd'] <= 35), 'age_category'] = 6


#年龄、年龄范围处理完后，删除不用特征
df_train.drop(['bd','age_range'],axis=1,inplace=True)
df_test.drop(['bd','age_range'],axis=1,inplace=True)

df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')


#--------------------------模型训练-----------------------------















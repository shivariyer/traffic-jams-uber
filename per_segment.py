import pandas as pd
from copy import deepcopy
import os
from joblib import Parallel, delayed

city = 'Sao_Paulo'
file_list = os.listdir('/scratch/ab9738/traffic/data/'+city+'/')
file_list = [file for file in file_list if '.csv' in file]

def save(df,way_id):
    df_way_id = deepcopy(df.loc[way_id])
    if isinstance(df_way_id, pd.Series):
        df_way_id.to_frame().transpose().set_index(['utc_timestamp'],drop=True)
    else:
        df_way_id = df_way_id.set_index(['utc_timestamp'],drop='True')
    df_way_id.to_csv('/scratch/ab9738/traffic/data/'+city+'/'+str(way_id)+'.csv',mode='a')

for file in file_list:
    df = pd.read_csv('/scratch/ab9738/traffic/data/'+city+'/'+file)

    df = df.drop(['year','month','day','hour','segment_id','start_junction_id',\
             'end_junction_id','osm_start_node_id','osm_end_node_id'],axis=1)

    df = df.set_index(['osm_way_id'])

    way_ids = list(set(df.index))

    Parallel(n_jobs=1)(delayed(save)(df,way_id) for way_id in [155587828])

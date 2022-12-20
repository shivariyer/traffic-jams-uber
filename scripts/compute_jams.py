import pandas as pd
from copy import deepcopy
import os
from joblib import Parallel, delayed
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm
from glob import glob
import pwlf
from multiprocessing import Manager
import warnings
warnings.filterwarnings('ignore')


def canbefloat(x):
    try:
        float(x)
    except:
        return False
    else:
        return True
    

def compute_and_write_segment_jams(test_file):
    ucanbefloat = lambda t: canbefloat(t)
    vfunc = np.vectorize(ucanbefloat)
    try:
        wayid = test_file.split(".")[0]
        test_file = '/scratch/ab9738/traffic/data/Nairobi/per-segment/'+test_file
        cols = pd.read_csv('/scratch/ab9738/traffic/data/Nairobi/movement-speeds-hourly-nairobi-2020-1.csv',nrows=1).columns
        df = pd.read_csv(test_file,header=None,on_bad_lines='skip',low_memory=False)
        df.columns = cols
        df = df.dropna()
        timestamp = df['utc_timestamp']
        timestamp = pd.to_datetime(timestamp, format="%Y-%m-%dT%H:%M:%S.000Z", errors='coerce')
        df = df.assign(utc_timestamp=timestamp)
        df = df.dropna()
        df = df.sort_values(by='utc_timestamp')
        tmp = df['speed_kph_mean'].to_numpy()
        df = df[vfunc(tmp)]

        series = tmp[vfunc(tmp)].astype(np.float)
        indices = np.cumsum(np.random.poisson(8, int(len(series)/8)))
        indices = indices[indices<len(series)]
        speed_sample = series[indices]

        speed_sample.sort()
        speed_sample = speed_sample[::-1]

        ind = np.arange(len(speed_sample))
        ind = ind/len(speed_sample)

        pwlf_func = pwlf.PiecewiseLinFit(ind, speed_sample)
        breaks = pwlf_func.fit(3, x_c=[ind[0],ind[-1]], y_c=[speed_sample[0],speed_sample[-1]])
        x_lin = np.linspace(ind[0], ind[-1], 1000)
        y_lin = pwlf_func.predict(x_lin)
        s_vals = pwlf_func.predict(breaks)
        s1, s2 = s_vals[2], s_vals[1]
        jam_th = (s1+s2)/4
        jams_df_ = df[series<jam_th]
        jams_df_ = jams_df_[['osm_way_id', 'utc_timestamp', 'speed_kph_mean']]

        if(len(jams_df_)):
            jams_df_ = jams_df_.reset_index()
            df_copy_ = deepcopy(jams_df_)
            i = 1
            while(i != len(jams_df_)):
                if(jams_df_.iloc[-i]['utc_timestamp']-pd.Timedelta(hours=1)==jams_df_.iloc[-i-1]['utc_timestamp']):
                    df_copy_ = df_copy_.drop(len(jams_df_)-i)
                i = i+1
            jams_df_ = df_copy_
        jams_df_.to_csv('/scratch/ab9738/traffic/traffic-jams-uber/results/jams_nairobi/'+wayid+'.csv', index=False)
        return(1)
    except:
        return(0)
    
def main():

    files = os.listdir('/scratch/ab9738/traffic/data/Nairobi/per-segment/')
    
    _ = Parallel(n_jobs=12)(delayed(compute_and_write_segment_jams)(test_file) for test_file in files)
        


if __name__ == "__main__":
    main()
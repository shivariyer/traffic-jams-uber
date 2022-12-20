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
np.random.seed(17)


def canbefloat(x):
    try:
        float(x)
    except:
        return False
    else:
        return True
    

def compute_and_write_segment_speeds(test_file):
    ucanbefloat = lambda t: canbefloat(t)
    vfunc = np.vectorize(ucanbefloat)
    wayid = test_file.split(".")[0]
    test_file = '/scratch/ab9738/traffic/data/NYC/per-segment/'+test_file
    cols = pd.read_csv('/scratch/ab9738/traffic/data/NYC/movement-speeds-hourly-new-york-2018-1.csv',nrows=1).columns
    df = pd.read_csv(test_file,header=None,on_bad_lines='skip',low_memory=False)
    df.columns = cols
    df = df.dropna()
    timestamp = df['utc_timestamp']
    timestamp = pd.to_datetime(timestamp, format="%Y-%m-%dT%H:%M:%S.000Z", errors='coerce')
    df = df.assign(utc_timestamp=timestamp)
    df = df.dropna()
    df = df.sort_values(by='utc_timestamp')
    tmp = df['speed_mph_mean'].to_numpy()
    if(len(df)>200):
        df = df[vfunc(tmp)]

        series = tmp[vfunc(tmp)].astype(np.float)
        indices = np.cumsum(np.random.poisson(8, int(len(series)/8)))
        indices = indices[indices<len(series)]
        speed_sample = series[indices]

        speed_sample.sort()
        speed_sample = speed_sample[::-1]

        ind = np.arange(len(speed_sample))
        ind = ind/len(speed_sample)
    
        if(len(speed_sample)>20):
            pwlf_func = pwlf.PiecewiseLinFit(ind, speed_sample, seed=17)
            breaks = pwlf_func.fit(3, x_c=[ind[0],ind[-1]], y_c=[speed_sample[0],speed_sample[-1]])
            s_vals = pwlf_func.predict(breaks)
            s1, s2 = s_vals[2], s_vals[1]
            return([wayid,s1,s2])
    
def main():

    files = os.listdir('/scratch/ab9738/traffic/data/NYC/per-segment/')
    row_values = Parallel(n_jobs=12)(delayed(compute_and_write_segment_speeds)(test_file) for test_file in files)
    row_values = [x for x in row_values if x is not None]
    row_arr = np.array(row_values)
    speeds_df = pd.DataFrame(row_arr, columns=['osm_wayid', 's1', 's2'])
    speeds_df.to_csv('/scratch/ab9738/traffic/traffic-jams-uber/results/speeds_nyc.csv', index=False)


if __name__ == "__main__":
    main()
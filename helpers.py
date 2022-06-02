#!/usr/bin/env python3

import sys
import os
import argparse

import subprocess
import psycopg2 as pg
import pandas.io.sql as psql

import pandas as pd
import numpy as np
from io import StringIO
from dateutil import tz
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

# Constants
startDate = pd.to_datetime("2018-01-01 00:00:00 +00:00")
endDate   = pd.to_datetime("2019-12-31 23:00:00 +00:00")

def rgDataFrame(wayId, fileName):
    connection = pg.connect("host=localhost dbname=traffic user=postgres password=1234")
    df = psql.read_sql('SELECT * FROM public.uber u where u.osm_way_id='+str(wayId) , connection)
    df = df[['utc_timestamp','osm_way_id','osm_start_node_id','osm_end_node_id','speed_kph_mean','speed_kph_stddev']]
    df = df[df['osm_way_id'] == int(wayId)]
    df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'],utc=True)
    df = df[['utc_timestamp','speed_kph_mean']]
    df = df.groupby(['utc_timestamp'],as_index=True).mean()
    if '2018' not in fileName and '2019' not in fileName:
        df.to_pickle('../pickles/dataframe-'+str(wayId)+'.pkl')
    return df


def reIndex(df):
    dr = pd.date_range(startDate, endDate, freq="H",tz = "UTC")
    df = df.reindex(dr)
    return df

def getDataframe(wayId, fileName = '../data/movement-speeds-hourly-nairobi.csv'):
    filePath = '../pickles/dataframe-'+str(wayId)+'.pkl'
    if(os.path.isfile(filePath)):
        return pd.read_pickle(filePath)
    else:
        print('error df not present'+str(wayId))
        return rgDataFrame(wayId,fileName)

def mkdirp(dirPath):
    if not os.path.isdir(dirPath):
        print('The directory is not present. Creating a new one..')
        os.mkdir(dirPath)
    else:
        print('The directory is present.')


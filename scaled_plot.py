#!/usr/bin/env python3

import sys
import os
import argparse

import subprocess

import pandas as pd
import numpy as np
from io import StringIO
from dateutil import tz
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

import helpers as h

def scaledPlotHelper(speed,s_low,s_high,x_low,x_high):
    speed_range = speed[speed['speed_kph_mean'].between(s_low,s_high)]
    speed_range = speed_range.reset_index().drop('index',1)
    speed_range['index'] = np.linspace(x_low,x_high,len(speed_range))
    speed_range.set_index('index',inplace=True)
    return speed_range

def getSpeed(wayId, df = None):
    if df is None:
        df = h.getDataframe(wayId)
    speed = df['speed_kph_mean'].reset_index().drop('utc_timestamp',1)
    speed = speed.sort_values(by='speed_kph_mean',ascending=False).reset_index().drop('index',1)
    return speed


def scaledPlot(s_lim,wayId):
    speed = getSpeed(wayId)
    s_max = max(speed['speed_kph_mean'])
    s_min = min(speed['speed_kph_mean'])
    s1 = 2.0*s_lim/3.0
    s2 = 0.5*s_lim
    speed_range_1 = scaledPlotHelper(speed,s_lim,s_max,0,0.2)
    speed_range_2 = scaledPlotHelper(speed,s1,s_lim ,0.2,1.0/3.0)
    speed_range_3 = scaledPlotHelper(speed,s2,s1,1.0/3.0,0.5)
    speed_range_4 = scaledPlotHelper(speed,s_min,s2 ,0.5,1)
    speed_range = pd.concat([speed_range_1,speed_range_2, speed_range_3,speed_range_4])
    speed_range = speed_range.groupby('index',as_index=True).speed_kph_mean.mean()
    speed_range.plot()
    plt.xlabel('Index')
    plt.ylabel('Speed (in kph)')
    plt.title('Scaled Decreasing Speed Plot')
    plt.legend(['Way ' + str(wayId)])
    plotName = '../plots/dec-speed-plots/scaled-'+ str(wayId)
    plt.savefig(plotName)
    return speed_range

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--wayid',type=int,
                        help = "Segment Osm way id", required=True)
    parser.add_argument('-l','--speedlim',type=int,
                        help = "Speed Limit for the way", required=True)
    args = parser.parse_args()

    scaledPlot(args.speedlim,args.wayid)


if __name__ == "__main__":
    sys.exit(main())

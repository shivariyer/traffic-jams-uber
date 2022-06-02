#!/usr/bin/env python3

import sys
import os
import argparse

import subprocess

import pandas as pd
import seaborn as sns
import numpy as np
from io import StringIO
from dateutil import tz
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

import matplotlib.dates as mdates
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import OrderedDict
import helpers as h
import scaled_plot as s
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

srcNo = {}
def plotPhases(wayId,fil_df,s_lim,ax,curr_style,noSpeedLabel = True):
    speed = s.getSpeed(wayId)
    s_max = max(speed['speed_kph_mean'])
    s_min = min(speed['speed_kph_mean'])
    s1 = 2.0*s_lim/3.0
    s2 = 0.5*s_lim
    print('s1:',s1)
    print('s2:',s2)
    # range 2: smax -> s1
    # range 1: s1 -> s2
    # range 0: s2 -> s_min

    fil_df['phase'] = fil_df.apply(lambda row: 2 if row['speed_kph_mean'] > s1 else(1 if row['speed_kph_mean'] > s2 else 0),axis=1)

    fil_df['date'] = pd.to_datetime(fil_df.index)
    grouped = fil_df.groupby([(fil_df.phase != fil_df.phase.shift()).cumsum()])
    # grouped = fil_df.groupby(fil_df['phase'].shift() != fil_df['phase'])
    # grouped = fil_df.groupby('phase')
    # colors = ['r','y','b','g']
    styles= ['-','--', '-.']
    legendVals = ['phase 0','phase 1','phase 2']
    fil_df['speed_kph_mean'].plot(ax=ax,color=curr_style, linestyle = ':',label = '_nolegend_')
    legends = []
    # legends.append(str(wayId) + ' Transition')
    for key, group in grouped:
        phaseVal = group['phase'][0]
        lbl = '$'+ srcNo[wayId] + '$ ' + legendVals[phaseVal]
        if lbl in legends:
            lbl = '_nolegend_'
        else:
            legends.append(lbl)
        group.plot(ax=ax, x="date", y="speed_kph_mean", linestyle=styles[phaseVal], linewidth=3,
                label = lbl , color = curr_style)

    if not noSpeedLabel:
        legends.append('$S_1$')
        legends.append('$S_2$')

    # plt.legend(list(OrderedDict.fromkeys(legends)))
    # handles, labels = plt.gca().get_legend_handles_labels()
    # by_label = OrderedDict(zip(labels, handles))

    
    plt.xlabel('Timestamp', labelpad=5 )
    plt.ylabel('Average Speed(in kph)', labelpad=5)
    ax.axhline(s1, color='indigo', ls='--',label= '_nolegend_' if noSpeedLabel else 'S1')
    ax.axhline(s2, color='aqua', ls='--',label= '_nolegend_' if noSpeedLabel else 'S2')
# plt.legend(by_label.values(), by_label.keys())
# plt.legend(list(OrderedDict.fromkeys(legends)))
    return legends


def plotDailyPhases(wayId,date, s_lim, ax,curr_style,noSpeedLabel = True):
    df = h.getDataframe(wayId)
    df = h.reIndex(df)
    fil_df = df.loc[date]
    plt.title('Phase change for '+ str(date))
    return plotPhases(wayId,fil_df,s_lim,ax,curr_style,noSpeedLabel)
# pltName = '../plots/phases/daily/' + str( wayId ) + ' ' + str(date)
    # plt.savefig(pltName)
    # return fil_df

def plotWeeklyPhases(wayId,date, s_lim, ax,curr_style,noSpeedLabel = True):
    startDate = pd.to_datetime(date,utc=True)
    endDate   = startDate + timedelta(weeks=1)
    # TODO: modify h.getDataframe to speed filter by wayId
    df = h.getDataframe(wayId)
    df = h.reIndex(df)
    fil_df = df.loc[startDate:endDate]
    plt.title('Phase change from '+ startDate.strftime("%d %b %y") + ' to ' + endDate.strftime("%d %b %y"))
    res = plotPhases(wayId,fil_df,s_lim,ax,curr_style,noSpeedLabel)
    plt.xlim(startDate,endDate)
    return res
    # pltName = '../plots/phases/weekly/' + str( wayId ) + ' ' + str(date)
    # plt.savefig(pltName)
    # return fil_df
def plotMonthlyPhases(wayId,date, s_lim, ax,curr_style,noSpeedLabel = True):
    startDate = pd.to_datetime(date,utc=True)
    endDate   = startDate + timedelta(days=30)

    df = h.getDataframe(wayId)
    df = h.reIndex(df)
    fil_df = df.loc[startDate:endDate]
    plt.xlim(startDate,endDate)

    plt.title('Phase change from '+ startDate.strftime("%d %b %y") + ' to ' + endDate.strftime("%d %b %y"))
    res = plotPhases(wayId,fil_df,s_lim,ax,curr_style, noSpeedLabel)
    return res
    # pltName = '../plots/phases/monthly/' + str( wayId ) + ' ' + str(date)
    # plt.savefig(pltName)
    # return fil_df


def plotList(ids,daily,weekly,monthly,ax, curr_style2, resetNoSpeedLabel = False):
    legends = []
    for indx in range(len(ids)):
        curr_style = curr_style2[indx]
        segmentId  = int(ids[indx])
        noSpeedLabel = True 
        if(resetNoSpeedLabel and indx == len(ids)-1):
            noSpeedLabel = False 
        print('segment Id: ',segmentId)
        speedLimit = 50
        if daily is not None:
            print('daily is not none:',daily)
            legends += plotDailyPhases(segmentId,daily,speedLimit,ax,curr_style)
        elif weekly is not None:
            print('weekly is not none:',weekly)
            legends = legends+ plotWeeklyPhases(segmentId,weekly,speedLimit,ax,curr_style, noSpeedLabel)
        elif monthly is not None:
            print('monthly is not none:',monthly)
            legends += plotMonthlyPhases(segmentId,monthly,speedLimit,ax,curr_style)

    # plt.legend(list(OrderedDict.fromkeys(legends)))
    return legends

class Junction:
    def __init__(self,name,src,dest):
        self.name  = name
        self.src   = src
        self.dest  = dest

def main():
    junctions = []
    # junctions.append(Junction('a' ,[  '344477771','24009413'  ],[  '4742016'  ]))
    # junctions.append(Junction('b' ,[  '41294345' ,'580233745'  ],[  '580233744'  ]))
    # junctions.append(Junction('c' ,[  '678371497','24025850','678371492','678371495' ],
    #                           ['555497908', '678371493', '678371494', '678371496']))
    junctions.append(Junction('d' ,[  '680030490' ,'680332724' ],[  '364279376'  ]))
    # junctions.append(Junction('e' ,[  '680332720' ,'680332719' ],[  '658240396'  ]))
    # junctions.append(Junction('e' ,['680030490','680332724'],['364279376']))
    # junctions.append(Junction('f', ['43763644','112950524','4724017','25606246'],['4724017','39573541','336067605','9931279']))


    for j in junctions:
        srcids = j.src
        destids = j.dest
        print('srcid:')
        print(j.src)
        print('destid:')
        print(j.dest)
        daily = monthly = None
        startDate = pd.to_datetime("2018-01-01 00:00:00 +00:00")
        endDate   = pd.to_datetime("2019-12-31 23:00:00 +00:00")
        for w in pd.date_range(startDate, endDate, freq="W",tz = "UTC"):
            weekly = w.strftime('%d-%m-%Y')
            plt.rcParams.update({'font.size': 26})
            fig, ax = plt.subplots(figsize=(16,9))

            curr_style = ['red','blue','green','gold']
            # print(args.srcids)
            # print(args.srcids[0])
            legends= []

            for i in range(len(srcids)):
                s = int(srcids[i])
                srcNo[s] = 's_' + str(i+1)

            for i in range(len(destids)):
                s = int(destids[i])
                srcNo[s] = 'd_' + str(i+1)
            print(srcNo)
            legends+= plotList(srcids,daily,weekly,monthly, ax, curr_style)
            # curr_style = 'blue'
            # plotList([srcids[1]],daily,weekly,monthly, ax, curr_style)
            curr_style = ['black','darkslategrey','brown','maroon']
            legends += plotList(destids,daily,weekly,monthly, ax, curr_style, True)
            print('Legends 2')
            print(list(OrderedDict.fromkeys(legends)))
            ax.tick_params(direction='out', top=0, right=0, length=10, pad=2)
            fig.legend(list(OrderedDict.fromkeys(legends)),ncol = 6,
                       loc='upper center', fontsize = 'x-small')

            ax.get_legend().remove()
            # plt.legend(list(OrderedDict.fromkeys(legends)), ncol = 4,
            #         bbox_to_anchor=(0,1.02,1,0.2), borderaxespad=0, fontsize = 'x-small')
            # plt.tight_layout()
            plt.subplots_adjust(bottom=0.18, left=0.08, right=0.95, top=0.75)
            # plt.legend(legends)
            # plt.show()
            name = 'junction_'+ j.name + "/"+ weekly
            plt.savefig('../plots/final/' + name)



if __name__ == "__main__":
    sys.exit(main())

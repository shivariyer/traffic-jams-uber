#!/usr/bin/env python3

import os
import sys

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from tqdm import tqdm


def plot_period(way_id, src_or_dst, start_dt, end_dt, ax, c=None, label='_'):

    fpath = os.path.join(datadir, '{}.csv'.format(way_id))
    if not os.path.exists(fpath):
        #print(fpath, 'does not exist.')
        return

    df = pd.read_csv(fpath, usecols=[1,2], index_col=[0], parse_dates=True)

    # the column is wrongly marked as "utc_timestamp" and the
    # timestamp values themselves have the UTC tz info wrongly
    # formatted into the string, when in fact the timestamps are local
    # times. That's why we are removing the UTC timestamps below and
    # renaming the index column.
    df = df.tz_localize(None, copy=False)
    df.index.rename('timestamp')

    # select only the week that we need
    df = df.loc[start_dt:end_dt]

    if src_or_dst == 'src':
        alpha = 0.5
        lwidth = 1.0
    elif src_or_dst == 'dst':
        alpha = 1.0
        lwidth = 3.0
    else:
        raise Exception('Unexpected value for src_or_dst')

    if way_id in table_final.index:
        s1, s2 = table_final.loc[way_id, ['s1','s2']]
    else:
        s1, s2 = 20, 40

    df.speed_kph_mean.where((df.speed_kph_mean < s1)).plot(ax=ax, ls='-', lw=lwidth, c='r', label='_', alpha=alpha)
    df.speed_kph_mean.where(((df.speed_kph_mean >= s1) & (df.speed_kph_mean < s2))).plot(ax=ax, ls='--', lw=lwidth, c='#A0A0A0', label='_', alpha=alpha)
    df.speed_kph_mean.where((df.speed_kph_mean >= s2)).plot(ax=ax, ls=':', lw=lwidth, c='#A0A0A0', label=label, alpha=alpha)

    return


class Junction:
    def __init__(self,name,src,dest):
        self.name  = name
        self.src   = src
        self.dest  = dest


if __name__ == "__main__":

    datadir = '/home/shivar/Research/traffic/traffic_jams/data/per-segment/'

    plt.rc('font', size=24)
    plt.rc('ps', useafm=True)
    plt.rc('pdf', use14corefonts=True)

    table_final = pd.read_csv('speed_cdf_stats_select_final.csv', index_col=0)
    
    # s_lim = 50
    # s_1 = 2*s_lim/3
    # s_2 = s_lim/2
    # s_2 = 40
    # s_1 = 20

    junctions = []

    junctions.append(Junction('junction_A' ,[  '344477771','24009413'  ],[  '4742016'  ]))
    junctions.append(Junction('junction_B' ,[  '41294345' ,'580233745'  ],[  '580233744'  ]))
    junctions.append(Junction('junction_C' ,[  '678371497','24025850','678371492','678371495' ],
                              ['555497908', '678371493', '678371494', '678371496']))
    junctions.append(Junction('junction_D' ,[  '680030490' ,'680332724' ],[  '364279376'  ]))
    #junctions.append(Junction('e' ,[  '680332720' ,'680332719' ],[  '658240396'  ]))
    #junctions.append(Junction('junction_E' ,['680030490','680332724'],['364279376']))
    junctions.append(Junction('junction_F', ['43763644','112950524','4724017','25606246'],['4724017','39573541','336067605','9931279']))

    # get the cluster centroids
    for folder in ['junction_k3_high', 'junction_k3_medium', 'junction_k3_low']:

        with open('../clustering/{}/list.txt'.format(folder)) as fin:
            for line in fin:
                parts = line.split(':')
                key = parts[0].strip()
                if key == 'in':
                    in_segs = list(map(int, parts[1].split(',')))
                elif key == 'out':
                    #out_seg = eval(parts[1])
                    out_segs = list(map(int, parts[1].split(',')))

        junctions.append(Junction(folder, in_segs, out_segs))


    for j in junctions:
        
        print('junction: {}'.format(j.name))
        print('srcid: {}'.format(j.src))
        print('destid: {}'.format(j.dest))

        savedir = '../figures/chosen_junctions/{}/'.format(j.name)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        daily = monthly = None
        start_date = pd.to_datetime('2018-01-01 00:00:00')
        end_date   = pd.to_datetime('2019-12-31 23:00:00')

        for week_begin_dt in tqdm(pd.date_range(start_date, end_date, freq='W-MON')):
            week_end_dt = week_begin_dt + timedelta(weeks=1)

            fig, ax = plt.subplots(figsize=(12,6))

            for ii, way_id in enumerate(j.src):
                #plot_period(way_id, week_begin_dt, week_end_dt, ax, c='#A0A0A0')
                plot_period(way_id, 'src', week_begin_dt, week_end_dt, ax)

            #dest_colors = ['black','darkslategrey','brown','maroon']
            #dest_colors = ['blue', 'brown', 'maroon', 'cyan', 'magenta']
            for ii, way_id in enumerate(j.dest):
                #plot_period(way_id, week_begin_dt, week_end_dt, ax, c='b', label='Dest segment')
                #plot_period(way_id, week_begin_dt, week_end_dt, ax, c=dest_colors[ii%len(dest_colors)], label=r'$d_{}$'.format(ii+1))
                plot_period(way_id, 'dst', week_begin_dt, week_end_dt, ax)
                
            # ax.axhline(s_1, color='indigo', ls='--', lw=3, label=r'$s_1$')
            # ax.axhline(s_2, color='aqua', ls='--', lw=3, label=r'$s_2$')
            ax.set_xlabel('Time', labelpad=5)
            ax.set_ylabel('Speed (kph)', labelpad=5)

            #fig.legend(loc='upper center', ncol=len(j.dest)+2, fontsize='small', fancybox=True)

            ax.tick_params(direction='out', top=0, right=0, length=10, pad=2)

            #fig.subplots_adjust(left=0.15, bottom=0.28, top=0.87, right=0.97)
            fig.subplots_adjust(left=0.15, bottom=0.28, top=0.97, right=0.97)
            savename = week_begin_dt.strftime('%Y-%m-%d')
            fig.savefig(savedir + savename + '.pdf')
            fig.savefig(savedir + savename + '.png')

            plt.close(fig)
        #     break
        # break

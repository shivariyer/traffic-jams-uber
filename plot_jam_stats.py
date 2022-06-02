import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

def compute_jam_stats(save_suffix, interp_lim=None, s1_arg=None):
    # save_suffix: some suffix for saving output files (cannot be empty string)
    # interp_lim: 2 or 3 or something else (interpolation limit to be passed to pd.DataFrame.interpolate)
    # s1_arg: what value of s1 to use (if None, use s1 of each segment individually)
    
    assert len(save_suffix) > 0
    assert (interp_lim is None) or (isinstance(interp_lim, int) and (interp_lim > 0))
    assert (s1_arg is None) or (s1_arg > 0)

    fout = open('jam_stats_{}.csv'.format(save_suffix), 'w')
    fout.write('wayid,n_jams,total_hours,min_hours,max_hours,mean_hours,median_hours,mode_hours,modes_hour_of_day,modes_num_instances' + os.linesep)

    savedir = '../jam_stats_{}/'.format(save_suffix)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    t_index = pd.date_range(start='1/1/2018', end='1/1/2020', freq='1H', closed='left')
    
    s1 = s1_arg

    data_dir = '/home/shivar/Research/traffic/traffic_jams/data/per-segment/'
    for wayid in tqdm(table_final.index, desc='interp_lim: {}, s1: {}'.format(interp_lim, s1_arg)):
        #wayid = 48958915

        data = pd.read_csv(data_dir + '{}.csv'.format(wayid), usecols=[1,2], index_col=0, parse_dates=True)
        data = data.tz_localize(None, copy=False) # remove time zone info
        data = data.reindex(t_index).interpolate(method='linear', limit=interp_lim)

        if s1_arg is None:
            s1 = table_final.s1.loc[wayid] # use the s1 for this particular segment, rather than population average
            #s1 = 20 # this is a lower bound on s1
        data['jam'] = np.nan
        data.loc[data.speed_kph_mean < s1, 'jam'] = 1.0
        data.loc[data.speed_kph_mean >= s1, 'jam'] = 0.0
        #data['jam'] = (data.speed_kph_mean < s1) # this is the definition of a jam

        # now count the number of times the jam occurs
        jams = [] # list of tuples -- (# hours, start time of jam, end time of jam)
        #state = data.jam.iloc[0]
        #s_dt = data.jam.index[0] if state else None
        state = False
        s_dt = None
        for ts, val in data.jam.iteritems():
            if (not state) and val == 1.0:
                # entering a jam when jam bit is 1 and state is False
                s_dt = ts
                state = True
            elif state and ((val == 0.0) or np.isnan(val)):
                # exiting a jam when jam bit is 0 or NaN and state is True
                hh = (ts - s_dt).total_seconds() / 3600.0
                jams.append((s_dt, ts, hh))
                state = False

        if len(jams) == 0:
            continue

        jams_df = pd.DataFrame(jams, columns=['start_time', 'end_time', 'hours_in_jam'])
        jams_df.set_index('start_time', inplace=True)
        jams_df.to_csv(savedir + '{}.csv'.format(wayid))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        jams_df.hours_in_jam.hist(ax=ax, bins=np.arange(25))
        ax.set_xlabel('Hours in jam')
        ax.set_ylabel('Number of jam instances')
        fig.tight_layout()
        fig.savefig(savedir + '{}.png'.format(wayid), facecolor='white')
        plt.close(fig)

        # stats
        # wayid,n_jams,total_hours,min_hours,max_hours,mean_hours,median_hours,modes_hours,modes_hour_of_day,modes_num_instances
        total_hours = jams_df.hours_in_jam.sum()
        min_hours = jams_df.hours_in_jam.min()
        max_hours = jams_df.hours_in_jam.max()
        mean_hours = jams_df.hours_in_jam.mean()
        median_hours = jams_df.hours_in_jam.median()
        modes_hours = jams_df.hours_in_jam.mode().to_list()
        modes_hour_of_day = stats.mode(jams_df.index.hour)
        fout.write('{},{},{:.0f},{:.0f},{:.0f},{:.1f},{:.1f},\"{}\",\"{}",\"{}"'.format(wayid, len(jams), total_hours, min_hours, max_hours, mean_hours, median_hours, modes_hours, modes_hour_of_day.mode, modes_hour_of_day.count) + os.linesep)

    fout.close()
    return



def print_plot_stats(fpath):
    table = pd.read_csv(fpath, index_col=0)
    table.sort_values('total_hours', inplace=True, ascending=False)
    
    # expecting file name to be of the form "jam_stats_xyz.csv" (xyz is the suffix being extracted)
    suffix = os.path.basename(os.path.splitext(fpath)[0]).split('_', 2)[2]
    fig_save_suffix = suffix.replace('_', '-') # for latex
    fout = open(os.path.splitext(fpath)[0] + '_report.txt', 'w')
    
    # total hours in jam
    total_total_hours = table.total_hours.sum()
    print('Total number of hours in jams by all the traffic in Nairobi: {}'.format(total_total_hours), file=fout, flush=True)
    print('Number of hours in jams by the traffic per day in Nairobi: {:.1f}'.format(total_total_hours/730.0), file=fout, flush=True)
    mean_total_hours = table.total_hours.mean()
    title_str = 'On avg {:.1f} hours over 2 years = {:.1f} hours per day'.format(mean_total_hours, mean_total_hours/730.0)
    print(title_str, file=fout, flush=True)

    plt.figure(figsize=(10,6))
    table.total_hours.hist(bins=20)
    plt.title(title_str)
    plt.xlabel('Hours', labelpad=10)
    plt.ylabel('Bin count', labelpad=10)
    plt.tight_layout()
    plt.savefig('../figures/total-jam-hours-{}.pdf'.format(fig_save_suffix))
    plt.close()
    
    # min, mean, median and max hours in jam- TODO
    plt.figure(figsize=(10,6))
    table.max_hours.hist(bins=20)
    plt.title('Max number of hours in jams')
    plt.xlabel('Hours', labelpad=10)
    plt.ylabel('Bin count', labelpad=10)
    plt.tight_layout()
    plt.savefig('../figures/max-jam-hours-{}.pdf'.format(fig_save_suffix))
    plt.close()
    
    # mode hours in jam
    mode_hours_all = sum(map(eval, table.mode_hours.values), [])
    mode_hours_bincount = np.bincount(mode_hours_all)
    print('Mode hours bincount:', file=fout)
    barx = np.arange(len(mode_hours_bincount))
    np.savetxt(fout, np.hstack((barx[:,np.newaxis], mode_hours_bincount[:,np.newaxis])), delimiter=',', fmt='%d')
    fout.flush()
    plt.figure(figsize=(8,4))
    plt.bar(np.arange(len(mode_hours_bincount)), mode_hours_bincount, color='m')
    plt.xlabel('Most frequently occurring jam durations', labelpad=10)
    plt.ylabel('Histogram', labelpad=10)
    plt.tight_layout()
    plt.savefig('../figures/mode-hours-bincount-{}.pdf'.format(fig_save_suffix))
    plt.close()
    
    # mode hour of day & number of instances
    mode_hourofday_all = sum(map(eval, table.modes_hour_of_day), [])
    mode_hourofday_count_all = sum(map(eval, table.modes_num_instances), [])
    mode_hourofday_df = pd.DataFrame(np.vstack((mode_hourofday_all, mode_hourofday_count_all)).T, 
                                     columns=['mode_hour_of_day', 'mode_hour_of_day_count'])
    mode_hourofday_counttotal_df = mode_hourofday_df.groupby('mode_hour_of_day').sum()
    mode_hourofday_counttotal_df = mode_hourofday_counttotal_df.reindex(np.arange(24))

    plt.figure(figsize=(10,5))
    mode_hourofday_counttotal_df.mode_hour_of_day_count.plot(kind='bar', color='c')
    print('Mode hour of day:', file=fout)
    np.savetxt(fout, mode_hourofday_counttotal_df.reset_index().values, delimiter=',', fmt='%.0f')
    plt.xlabel('Hour of day', labelpad=10)
    plt.ylabel('Number of occurrences', labelpad=10)
    plt.tight_layout()
    plt.savefig('../figures/mode-hourofday-{}.pdf'.format(fig_save_suffix))
    plt.close()
    
    fout.close()


if __name__ == '__main__':
    
    # compute traffic jam stats
    plt.rc('font', size=20)
    plt.rc('ps', useafm=True)
    plt.rc('pdf', use14corefonts=True)

    # only use the table_final wayids
    table_final = pd.read_csv('speed_cdf_stats_select_final.csv', index_col=0)

    #compute_jam_stats('interplim_2', interp_lim=2)
    #compute_jam_stats('interplim_2_s1_20', interp_lim=2, s1_arg=20)
    #compute_jam_stats('interplim_3', interp_lim=3)
    #compute_jam_stats('interplim_3_s1_20', interp_lim=3, s1_arg=20)

    print_plot_stats(sys.argv[1])

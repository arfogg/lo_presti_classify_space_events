# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:07:24 2025

@author: A R Fogg
"""

import sys
import os
import pathlib
import string
import matplotlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(r'C:\Users\admin\Documents\wind_waves_akr_code\readers')
import read_omni
import read_supermag

alphabet = list(string.ascii_lowercase)
axes_labels = []
for a in alphabet:
    axes_labels.append('(' + a + ')')

def read_supermag_csv():

    print('Reading SuperMAG csv file')

    # Have file in wdir
    supermag_csv = '2012-06-16.csv'

    print('Reading SuperMAG data from: ', supermag_csv)

    df = pd.read_csv(supermag_csv, delimiter=',', parse_dates=['Date_UTC'],
                     float_precision='round_trip')
 
    return df



def summary_plot():
    
    
    sc_onset = pd.Timestamp('2012-06-16 20:19:00')

    

    # Read in SMR
    indices_df = read_supermag.read_indices_annual(sc_onset.year)
    inddf_time_filter = (indices_df['Date_UTC'] >= 
                         sc_onset - pd.Timedelta(minutes=20)) & \
                        (indices_df['Date_UTC'] <=
                         sc_onset + pd.Timedelta(minutes=20))
    indices_df = indices_df.loc[inddf_time_filter].reset_index(drop=True)

    # Read in SuperMAG magnetometer data
    magn_df = read_supermag_csv()

    # Filter by SC +/- onset
    magdf_time_filter = (magn_df['Date_UTC'] >=
                         sc_onset - pd.Timedelta(minutes=20)) & \
                        (magn_df['Date_UTC'] <=
                         sc_onset + pd.Timedelta(minutes=20))
    magn_df = magn_df.loc[magdf_time_filter].reset_index(drop=True)

    # Filter by MLT
    maddf_mlt_filter = ((magn_df['MLT'] >= 11.5) & (magn_df['MLT'] <= 12.5))
    magn_df = magn_df.loc[maddf_mlt_filter].reset_index(drop=True)

    # Stations available
    # stations = np.array(magn_df.IAGA.unique())
    
    # For speed of getting draft done, manually keeping stations without big
    # bits of missing data
    stations = np.array(['T58', 'T25'])

    # Number of stations
    n_stations = stations.size
    
    # Find average CoLat for each station
    
    stat_ave_colat = np.full(n_stations, np.nan)
    for i in range(n_stations):       
        stat_ave_colat[i] = np.nanmean(magn_df.loc[magn_df.IAGA == stations[i], 'MCOLAT'])
        
        
        
    # Order the stations by CoLat (lowest/most polar first)
    stations = stations[np.argsort(stat_ave_colat)]
    stat_ave_colat = stat_ave_colat[np.argsort(stat_ave_colat)]

    # Formatting definitions
    fontsize = 12
    mag_fmt_dict = {'color': 'mediumorchid',
                    'linewidth': 1.}
    SMR_fmt_dict = {'color': 'black',
                    'linewidth': 1.,
                    'label': 'SMR'}
    vline_fmt_dict = {'color': 'grey',
                      'linewidth': 1.5,
                      'label': 'onset',
                      'linestyle': 'dashed'}
    
    
    fig, ax = plt.subplots(nrows=n_stations + 1,
                           figsize=(8, (n_stations + 1) * 2))
    
    for (i, stat) in enumerate(stations):
        # Subset data by station
        stat_df = magn_df.loc[magn_df.IAGA == stations[i]]
        
        # Plot station data
        ax[i].plot(stat_df.Date_UTC, stat_df.dbn_nez, **mag_fmt_dict)

        # Add text with station name, lat, MLT
        onset_df = stat_df.loc[stat_df.Date_UTC == sc_onset]
        stat_label = str(stations[i]) + '\n' +\
                     str(onset_df.MCOLAT.values[0]) + '$^{\circ}$' + '\n' +\
                     str(onset_df.MLT.values[0]) + ' MLT'
        t = ax[i].text(0.95, 0.02, stat_label, transform=ax[i].transAxes,
                    fontsize=fontsize, va='bottom', ha='right',
                    color=mag_fmt_dict['color'])
        
        # Formatting
        ax[i].set_ylabel("$dBN_{nez}$ (nT)", fontsize=fontsize)
        ax[i].set_xticklabels([])        
        
     
    # Plot SMR    
    ax[-1].plot(indices_df.Date_UTC, indices_df.SMR, **SMR_fmt_dict)
    ax[-1].set_ylabel('SMR (nT)', fontsize=fontsize)
    ax[-1].set_xlabel('UT', fontsize=fontsize)
    ax[-1].xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%H:%M\n%d/%m/%y"))
    
    
    for j, a in enumerate(ax):
        

        #tick fontsize
        a.tick_params(labelsize=fontsize)
        a.set_xlim(sc_onset-pd.Timedelta(minutes=20), sc_onset+pd.Timedelta(minutes=20))
        
        t = a.text(0.02, 0.92, axes_labels[j], transform=a.transAxes,
                    fontsize=fontsize, va='top', ha='left')
        t.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='grey'))
        
        a.axvline(sc_onset, **vline_fmt_dict)
    
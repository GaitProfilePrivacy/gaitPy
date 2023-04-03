#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:06:01 2023

@author: ozgesurer
"""
import matplotlib.pyplot as plt
import pyreadr
import pandas as pd

result = pyreadr.read_r('../data/gp.rds')
df = result[None]
sample_subjects = ['02', '06', '10', '14']
sample_cycles = ['10', '15', '20', '25', '30', '35', '40']


df_select = df.loc[df['gait_cycle'].isin(sample_cycles) & df['subject_num'].isin(sample_subjects)]



for sub in sample_subjects:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    obs = df_select.loc[df['subject_num'].isin([sub])]
    obs.set_index('gait_time', inplace=True)

    obs.groupby('gait_cycle')['acc_magnitude'].plot(legend=False)
    plt.show()
        
dfs = pd.read_excel('DemoData.xlsx', sheet_name='Sheet1')    

# subject_mean = df.groupby(['subject_num', 'gait_cycle'])['acc_magnitude'].mean()

#sample_subjects = ['02', '06', '10', '14']

#subject_mean.groupby('subject_num').acc_magnitude.plot(legend=False)
#subject_mean.groupby('subject_num').reset_index()
#for sub in sample_subjects:
#    df_sub = df.loc[df['subject_num'].isin([sub])] 
#    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
#    ax.plot(df_sub['gait_cycle'])

#df.groupby(['subject_num', 'gait_cycle']).acc_magnitude.mean().plot()
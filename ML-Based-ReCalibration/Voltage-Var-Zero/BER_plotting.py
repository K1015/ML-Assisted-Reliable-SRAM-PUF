# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 04:58:28 2022

@author: Thermal Chamber
"""
import glob, os
import numpy as np
import re
import matplotlib.pyplot as plt
import copy
import os


path = ''
GoldenRespAmbient = np.load(path+'GResp_volt_Zero_ambient_Oct11.npy')
Reliability_PUF = np.load(path+'Rel_volt_Zero_ambient_Oct11.npy')
GoldenResp = np.load(path+'GResp_volt_Zero_all_Oct11.npy')
Volt_all = np.array([7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,
                     10,10.2,10.4,10.6,10.8,11,11.2,11.4,11.6,11.8,12])
nChal = 6815
respW = 32

BoardTotal = 8
BER = np.zeros([BoardTotal,len(Volt_all)])
refloc = np.where(Volt_all==7)[0][0]
for i in range(BoardTotal):
    for j in range(len(Volt_all)):
        BER[i,j] = np.sum(GoldenRespAmbient[i,:,:]!=GoldenResp[i,j,:,:])/(nChal*respW)*100
  
#%%

Window_Acc = np.load('Resp_Transfer_Window_Acc.npy')

Window_Acc = np.vstack([Window_Acc,np.load('Resp_Transfer_Window_Acc_Oct14.npy')])

best_board = 1
worst_board = 3

Window_Acc_specific = np.load('Resp_Continual_Window_Acc.npy')

Window_Acc_specific = np.vstack([Window_Acc_specific,np.load('Resp_Continual_Window_Acc_Oct14.npy')])
#%%
import matplotlib.pyplot as plt
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
#my_xticks = np.arange(24,60,2.5)
#plt.xticks(Volt_all, my_xticks)
plt.ylabel('Bit Error Rate in %',fontsize=42,fontdict=font)
plt.xlabel('Voltage in V',fontsize=42,fontdict=font)
for i in range(BoardTotal):
    ax.plot(Volt_all[::2], BER[i,::2],marker="o",markersize=10,label='%d'%(i+1))
ax.set_ylim([-0.5,5])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
# plt.text(7.25,5.5,'Boards',fontsize=28, fontfamily='Arial',fontweight="bold")

# plt.legend(loc='upper right', ncol=4,bbox_to_anchor=(1, 1.25),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax.grid(True)

plt.savefig('BER_SRAM_PUF_Voltage_Zero.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

#%%
ms = 12
fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
#my_xticks = np.arange(24,60,2.5)
#plt.xticks(Volt_all, my_xticks)
plt.ylabel('# of Window Errors \nPred Acc %',fontsize=42,fontdict=font)
plt.xlabel('Voltage in V',fontsize=42,fontdict=font)

ax.plot(Volt_all[1::3], Window_Acc[best_board,1::3],marker="o",markersize=ms,label='Continual Learning',color='green')
ax.plot(Volt_all[1::3], Window_Acc_specific[best_board,1::3],marker="o",markersize=ms,label='Transfer Learning',color='darkorange')
ax.plot(Volt_all[1::3], Window_Acc_specific[worst_board,1::3],marker="s",markersize=ms,label='PUF Specific Learning',color='yellowgreen')
ax.plot(Volt_all[1::3], Window_Acc[worst_board,1::3],marker="s",markersize=ms,label='Transfer Learning',color='darkgoldenrod')

ax.set_ylim([75,102])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
# plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.45, 1.18),prop={'size': 28, 'family':'Arial', 'weight':'regular'}                  
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Window_accuracy_prediction_volt_Zero.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

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
GoldenResp = np.load(path+'GResp_UNO_temp_all.npy')
Reliability_PUF = np.load(path+'Rel_UNO_AmbientTemp_nMeas15.npy')
Temp_all = np.arange(-23.5,70,2.5)
BER = np.zeros([10,len(Temp_all)])
refloc = np.where(Temp_all==24)[0][0]

for i in range(10):
    for j in range(len(Temp_all)):
        BER[i,j] = np.sum(GoldenResp[i,refloc,:,:]!=GoldenResp[i,j,:,:])/(128*64)*100    

Window_Acc_CL = np.load('WindowErrorPredAcc_UNO_ContinualLearning.npy')

Window_Acc_TL = np.load('WindowErrorPredAcc_UNO_TransferLearning.npy')
#%%
import matplotlib.pyplot as plt
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
plt.ylabel('Bit Error Rate in %',fontsize=36,fontdict=font)
plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=36,fontdict=font)
for i in range(10):
    ax.plot(Temp_all[1::3], BER[i,1::3],marker="o",markersize=10,label='%d'%(i+1))
ax.set_ylim([-1,20])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
#plt.text(42,18,'Boards',fontsize=28, fontfamily='Arial')

# plt.legend(loc='upper right', ncol=5,bbox_to_anchor=(1, 0.9),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax.grid(True)

plt.savefig('BER_SRAM_PUF_Temp.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

#%%
import matplotlib.pyplot as plt
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 

plt.ylabel('# of Window Errors \nPred Acc %',fontsize=36,fontdict=font)
plt.xlabel('Temperature in C',fontsize=36,fontdict=font)

Board_ID = 8

ax.plot(Temp_all[1::3], Window_Acc_CL[Board_ID,1::3],marker="o",markersize=15,label='CC1: Continual Learning',color='green')
ax.plot(Temp_all[1::3], Window_Acc_TL[Board_ID,1::3],marker="o",markersize=15,label='CC2: Transfer Learning',color='darkorange')

ax.set_ylim([70,102])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
# plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.51, 1.22),prop={'size': 28, 'family':'Arial', 'weight':'regular'}                  
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Window_accuracy_prediction_temp.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

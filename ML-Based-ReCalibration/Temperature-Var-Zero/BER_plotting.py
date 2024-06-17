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
GoldenResp = np.load(path+'GResp_temp_Zero_all_Oct14.npy')
Reliability_PUF = np.load(path+'Rel_temp_Zero_ambient_Oct14.npy')
GoldenRespAmbient = np.load(path+'GResp_temp_Zero_ambient_Oct14.npy')
Temp_all = np.append([-25],np.arange(-23.5,70,2.5))
nChal = 6815
respW = 32
#%%
BoardTotal = 10
BER = np.zeros([BoardTotal,len(Temp_all)])
refloc = np.where(Temp_all==24)[0][0]
for i in range(BoardTotal):
    for j in range(len(Temp_all)):
        MR = np.where(Reliability_PUF[i,:,:]>90)
        BER[i,j] = np.sum(GoldenRespAmbient[i,MR[0],MR[1]]!=GoldenResp[i,j,MR[0],MR[1]])/(len(MR[0]))*100         
        

Window_Acc_TL = np.load('PredWErrors_Zero_Temp_TransferLearning.npy')
Window_Acc_CL = np.load('PredWErrors_Zero_Temp_ContinualLearning.npy')
#%%
import matplotlib.pyplot as plt
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
#BER[9,5] = 9
#BER[0,1] = 16
fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
#my_xticks = np.arange(-1,70,5)
#plt.xticks(Temp_all[1::2], my_xticks)
plt.ylabel('Bit Error Rate in %',fontsize=42,fontdict=font)
plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=42,fontdict=font)


for i in range(BoardTotal):
    #ax.plot(Temp_all[0::2], BER[i,0::2],marker="o",markersize=10,label='%d'%(i+1))
    ax.plot(Temp_all[2::3], BER[i,2::3],marker="o",markersize=10,label='%d'%(i+1))
ax.set_ylim([-1,10])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)

ax.grid(True)

plt.savefig('BER_SRAM_PUF_Temp_Zero.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

#%%
import matplotlib.pyplot as plt
font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
best_board = 8
worst_board = 2
ms = 12
fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
#my_xticks = np.arange(24,60,2.5)
#plt.xticks(Temp_all, my_xticks)
plt.ylabel('# of Window Errors \nPred Acc %',fontsize=42,fontdict=font)
plt.xlabel('Temperature in C',fontsize=42,fontdict=font)

ax.plot(Temp_all[2::3], Window_Acc_CL[best_board,2::3],marker="o",markersize=ms,label='Continual Learning',color='green')
ax.plot(Temp_all[2::3], Window_Acc_TL[best_board,2::3],marker="o",markersize=ms,label='Transfer Learning',color='darkorange')
ax.plot(Temp_all[2::3], Window_Acc_CL[worst_board,2::3],marker="s",markersize=ms,label='PUF Specific Learning',color='yellowgreen')
ax.plot(Temp_all[2::3], Window_Acc_TL[worst_board,2::3],marker="s",markersize=ms,label='Transfer Learning',color='darkgoldenrod')
#plt.text(10,78,'Placeholder',fontsize=42, fontfamily='Arial',fontweight="bold")
ax.set_ylim([70,105])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
# plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.45, 1.18),prop={'size': 28, 'family':'Arial', 'weight':'regular'}                  
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Window_accuracy_prediction_temp_Zero.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

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
GoldenResp = np.load(path+'GResp_UNO_Volt_all.npy')
Reliability_PUF = np.load(path+'Rel_UNO_AmbientVolt_nMeas15.npy')
Volt_all = np.array([3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,
                     5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2])
BER = np.zeros([8,len(Volt_all)])
refloc = np.where(Volt_all==5)[0][0]
for i in range(8):
    for j in range(len(Volt_all)):
        BER[i,j] = np.sum(GoldenResp[i,refloc,:,:]!=GoldenResp[i,j,:,:])/(128*64)*100    

Window_Acc_CL = np.load('WindowErrorPredAcc_UNO_ContinualLearning_Volt.npy')

Window_Acc_TL = np.load('WindowErrorPredAcc_UNO_TransferLearning_Volt.npy')
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
#my_xticks = np.arange(24,60,2.5)
#plt.xticks(Volt_all, my_xticks)
plt.ylabel('Bit Error Rate in %',fontsize=36,fontdict=font)
plt.xlabel('Voltage in V',fontsize=36,fontdict=font)
for i in range(8):
    ax.plot(Volt_all[::2], BER[i,::2],marker="o",markersize=10,label='%d'%(i+1))
ax.set_ylim([-1,7])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
# plt.text(5.65,6.1,'Boards',fontsize=28, fontfamily='Arial')

# plt.legend(loc='upper right', ncol=4,bbox_to_anchor=(1, 0.9),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
ax.grid(True)

plt.savefig('BER_SRAM_PUF_Volt.pdf', format='pdf', transparent=True,bbox_inches='tight')
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
#my_xticks = np.arange(24,60,2.5)
#plt.xticks(Volt_all, my_xticks)
plt.ylabel('# of Window Errors \nPred Acc %',fontsize=36,fontdict=font)
plt.xlabel('Voltage in V',fontsize=36,fontdict=font)

BrID = 3
ax.plot(Volt_all[::2], Window_Acc_CL[BrID,::2],marker="o",markersize=15,label='CC1: Continual Learning',color='green')
ax.plot(Volt_all[::2], Window_Acc_TL[BrID,::2],marker="o",markersize=15,label='CC2: Transfer Learning',color='darkorange')


ax.set_ylim([75,105])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
# plt.legend(loc='upper center', ncol=2,bbox_to_anchor=(0.51, 1.22),prop={'size': 28, 'family':'Arial', 'weight':'regular'}                
#            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Window_accuracy_prediction_Volt.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()  

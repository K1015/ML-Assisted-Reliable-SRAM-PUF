#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:21:47 2022

"""
#Plot Entropy Loss
import numpy as np
import matplotlib.pyplot as plt

font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

Temp = np.arange(-25,71,5)

csfont = {'fontname':'Arial'}

from matplotlib import rc
rc('font',**{'family':'serif','serif':['Arial']})
path = 'Data_Files/'

U_pre_TransferL = np.load(path+'Uniformity_pre_TL.npy')
U_post_TransferL = np.load(path+'Uniformity_post_TL.npy')

MinEntropy = np.load(path+'minEntropy_pre_TL.npy')
MinEntropy_post_TransferL = np.load(path+'minEntropy_post_TL.npy')

MinEntropy_Loss_multiple_p = np.load(path+'minEntropy_loss_multiple_p.npy')
MinEntropy_Loss_fixed_p = np.load(path+'minEntropy_loss_fixed_p.npy')
MinEntropy_Loss_fixed_p_TransferL = np.load(path+'minEntropy_Loss_fixed_p_ECC_TL.npy')
#%%
fig, ax = plt.subplots(figsize=(9,9))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
my_xticks = np.arange(-25,71,20)
plt.xticks(Temp[::4], my_xticks)
ax2=ax.twinx()
ax.set_ylabel('Uniformity %',fontsize=48,fontdict=font,color='black')
ax.set_xlabel(r'Temperature in $^{\circ}$C',fontsize=48,fontdict=font)
ax2.set_ylabel('Min Entropy (bits)',fontsize=48,fontdict=font,color='black')
a = ax2.plot(Temp[1:], MinEntropy[1:],marker="d",markersize=12,linestyle='-',linewidth=3,color='purple',label=r'$H_{\infty}(\mathbf{x})$: Min-Entropy with ECC - BCH[15,11,1]')
b = ax2.plot(Temp[1:], MinEntropy_post_TransferL[1:],marker="s",markersize=12,linestyle='--',linewidth=3,color='olive',label=r'$H_{\infty}(\mathbf{x}_c)$: Min-Entropy with ECC + Transfer Learning')
c = ax.plot(Temp[1:], U_pre_TransferL[1:]*100,marker="d",markersize=12,linestyle='-',linewidth=3,color='darkorange',label='Uniformity with ECC - BCH[15,11,1]')
d = ax.plot(Temp[1:], U_post_TransferL[1:]*100,marker="s",markersize=12,linestyle='--',linewidth=3,color='teal',label='Uniformity with ECC + Transfer Learning')
ax.set_xlim([-30,75])
ax.set_ylim([45,76])
ax2.set_ylim([6,18])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax2.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
lns = a+b+c+d
labs = [l.get_label() for l in lns]
ax2.legend(loc=0)
plt.legend(lns,labs,loc='upper center', ncol=1,bbox_to_anchor=(0.49, 1.5),prop={'size': 32, 'family':'Arial'}         
            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Only_Min_Entropy_and_Uni.pdf', format='pdf', transparent=True,bbox_inches='tight',dpi=200)
plt.show()

#%%
fig, ax = plt.subplots(figsize=(9,9))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
my_xticks = np.arange(-25,71,20)
plt.xticks(Temp[::4], my_xticks)
plt.ylabel('Min Entropy Loss',fontsize=48,fontdict=font)
plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=48,fontdict=font)
ax.plot(Temp[1:], MinEntropy_Loss_multiple_p[1:],marker="v",markersize=12,color='dodgerblue',label=r'multiple helper data + ECC-BCH[15,11,1]')
ax.plot(Temp[1:], MinEntropy_Loss_fixed_p[1:],marker="d",markersize=12,color='red',label=r'fix helper data + ECC-BCH[15,11,1]')
ax.plot(Temp[1:], MinEntropy_Loss_fixed_p_TransferL[1:],marker="s",markersize=12,color='darkorange',label=r'fix helper data + ECC + Transfer Learning')
ax.set_xlim([-30,75])
ax.set_ylim([1,3.5])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
plt.legend(loc='best', prop={'size': 18, 'family':'Arial'})
plt.legend(loc='upper center', ncol=1,bbox_to_anchor=(0.48, 1.38),prop={'size': 32, 'family':'Arial'}         
            ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Min_Entropy_loss.pdf', format='pdf', transparent=True,bbox_inches='tight',dpi=200)
plt.show()

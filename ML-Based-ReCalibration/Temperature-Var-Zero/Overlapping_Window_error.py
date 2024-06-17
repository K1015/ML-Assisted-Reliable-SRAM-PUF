# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 04:58:28 2022

@author: Thermal Chamber
"""
import glob, os
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }
# Code for generating the number of errors all windows (including overlapping ones)
# Incldes all bits even the unreliable ones as info necessary to learn
# With respect to 25C golden response

path = '/Users/saionroy/Documents/Graduate_Acad/KP_Research/Sem-8/TCHES24/Temperature_variation_Zero/'
GoldenRespAmbient = np.load(path+'GResp_temp_ambient_Oct14.npy')
GoldenResp = np.load(path+'GResp_temp_all_Oct14.npy')
Temp_all = np.append([-25],np.arange(-23.5,70,2.5))

rng = 2 #Window Size, create non overlapping windows
pltrng = 5  

cnt = 0

nChal = 6815
respW = 32
R_overlap = nChal - rng + 1
C_overlap = respW - rng + 1
#Windows overlapping
BoardTotal = 10
#Windows overlapping
Window_errors_overlap = np.zeros([BoardTotal,len(Temp_all),R_overlap,C_overlap])
refloc = np.where(Temp_all==24)[0][0]

for board in range(1):
    for temp in range(len(Temp_all)):    
        for i in range(0,R_overlap):
            for j in range(0,C_overlap):
               Window_errors_overlap[board,temp,i,j] = (GoldenResp[board,temp,i:i+rng,j:j+rng]!=GoldenResp[board,refloc,i:i+rng,j:j+rng]).sum() 
                

outputPath = '/Users/saionroy/Documents/Graduate_Acad/KP_Research/Sem-8/TCHES24/Temperature_variation_Zero/'
#np.save(outputPath+'Window_error_overlapping_Oct14.npy',Window_errors_overlap)
#%%
Temp_val = 1
from matplotlib import pyplot as plt
fig, ax = plt.subplots(figsize=(7,7))
plt.yticks(fontname = "Arial") 
plt.xticks(fontname = "Arial") 
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
plt.imshow(Window_errors_overlap[0,Temp_val,:32*8,:].reshape([256,256]), interpolation='none')
plt.ylabel('rows',fontsize=36,fontdict=font)
plt.xlabel('columns',fontsize=36,fontdict=font)
#plt.savefig('error_pattern.pdf',transparent=True,bbox_inches='tight')
plt.savefig('error_pattern_Temp_%d.png'%(Temp_all[Temp_val]),transparent=True,bbox_inches='tight',dpi=200)
plt.show()
#%%
# for i in range(0,int(128/rng),5):
#     for j in range(0,int(64/rng),5): 
#         if(np.sum(Window_errors_overlap[0,:,i,j])!=0):
#             fig, ax = plt.subplots(figsize=(8,4))
#             plt.yticks(fontname = "Arial") 
#             plt.xticks(fontname = "Arial") 
#             #my_xticks = np.arange(-3.5,70,2.5)
#             plt.ylim(-1,4) 
#             #plt.xticks(Temp_all, my_xticks)
#             plt.ylabel('# of bits in error',fontsize=25,fontdict=font)
#             plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=25,fontdict=font)
#             ax.plot(Temp_all, Window_errors_overlap[0,:,i,j],marker="o",markersize=10)
#             #ax.set_ylim([0.02,0.06])
#             ax.xaxis.set_tick_params(labelsize=15)
#             ax.yaxis.set_tick_params(labelsize=15)
#             ax.grid(True)
            
#             #plt.savefig('Plots/Bit_Error_vs_Temp_v%d_%d.png'%(i,j), format='png', transparent=True,bbox_inches='tight')
#             plt.show()      

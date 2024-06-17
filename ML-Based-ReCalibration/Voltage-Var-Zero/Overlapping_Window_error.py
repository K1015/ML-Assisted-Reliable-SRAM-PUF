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

path = ''
GoldenRespAmbient = np.load(path+'GResp_volt_Zero_ambient_Oct14.npy')
GoldenResp = np.load(path+'GResp_volt_Zero_all_Oct14.npy')
Volt_all = np.array([7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,
                     10,10.2,10.4,10.6,10.8,11,11.2,11.4,11.6,11.8,12])
refloc = np.where(Volt_all==7)[0][0]
rng = 2 #Window Size, create non overlapping windows
pltrng = 5  

cnt = 0
nChal = 6815
respW = 32
R_overlap = nChal - rng + 1
C_overlap = respW - rng + 1
BoardTotal = 3
#Windows overlapping
Window_errors_overlap = np.zeros([BoardTotal,len(Volt_all),R_overlap,C_overlap])


for board in range(BoardTotal):
    for volt in range(len(Volt_all)):    
        for i in range(0,R_overlap):
            for j in range(0,C_overlap):
               Window_errors_overlap[board,volt,i,j] = (GoldenRespAmbient[board,i:i+rng,j:j+rng]!=GoldenResp[board,volt,i:i+rng,j:j+rng]).sum() 
                

outputPath = ''
np.save(outputPath+'Window_error_overlapping_2x2_volt_Zero_Oct14.npy',Window_errors_overlap)
   
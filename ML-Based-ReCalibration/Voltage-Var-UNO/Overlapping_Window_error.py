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
GoldenResp = np.load(path+'GResp_UNO_Volt_all.npy')
Reliability_PUF = np.load(path+'Rel_UNO_AmbientVolt_nMeas15.npy')
Volt_all = np.array([3.8,3.9,4,4.1,4.2,4.3,4.4,4.5,4.6,4.7,4.8,4.9,5,
                     5.1,5.2,5.3,5.4,5.5,5.6,5.7,5.8,5.9,6,6.1,6.2])
refloc = np.where(Volt_all==5)[0][0]
rng = 2 #Window Size, create non overlapping windows
pltrng = 5  

cnt = 0

R_overlap = 128 - rng + 1
C_overlap = 64 - rng + 1
#Windows overlapping
Window_errors_overlap = np.zeros([8,len(Volt_all),R_overlap,C_overlap])


for board in range(8):
    for temp in range(len(Volt_all)):    
        for i in range(0,R_overlap):
            for j in range(0,C_overlap):
               Window_errors_overlap[board,temp,i,j] = (GoldenResp[board,temp,i:i+rng,j:j+rng]!=GoldenResp[board,refloc,i:i+rng,j:j+rng]).sum() 
                

outputPath = ''
np.save(outputPath+'Window_error_overlapping_UNO_2x2_Volt.npy',Window_errors_overlap)

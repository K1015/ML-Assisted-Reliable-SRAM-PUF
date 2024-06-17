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
GoldenResp = np.load(path+'GResp_UNO_temp_all.npy')
Reliability_PUF = np.load(path+'Rel_UNO_AmbientTemp_nMeas15.npy')
Temp_all = np.arange(-23.5,70,2.5)

rng = 2 #Window Size, create non overlapping windows

cnt = 0

R_overlap = 128 - rng + 1
C_overlap = 64 - rng + 1
#Windows overlapping
Window_errors_overlap = np.zeros([10,len(Temp_all),R_overlap,C_overlap])
refloc = np.where(Temp_all==24)[0][0]

for board in range(10):
    for temp in range(len(Temp_all)):    
        for i in range(0,R_overlap):
            for j in range(0,C_overlap):
               Window_errors_overlap[board,temp,i,j] = (GoldenResp[board,temp,i:i+rng,j:j+rng]!=GoldenResp[board,refloc,i:i+rng,j:j+rng]).sum() 
                

outputPath = ''
np.save(outputPath+'Window_error_overlapping_UNO_2x2.npy',Window_errors_overlap)
   
  

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 22:59:06 2022

"""
import numpy as np
import json

from itertools import chain

    
with open("Codewords_w.json") as json_file:
    Codewords_w = json.load(json_file)
    
    
with open("std_array_p.json") as json_file:
    std_array_p = json.load(json_file)
    

std_array_p[0].append('100000000000000')
std_array_p[1].append('010000000000000')
std_array_p[2].append('001000000000000')
std_array_p[3].append('000100000000000')
std_array_p[4].append('000010000000000')
std_array_p[5].append('000001000000000')
std_array_p[6].append('000000100000000')
std_array_p[7].append('000000010000000')
std_array_p[8].append('000000001000000')
std_array_p[9].append('000000000100000')
std_array_p[10].append('000000000010000')
std_array_p[11].append('000000000001000')
std_array_p[12].append('000000000000100')
std_array_p[13].append('000000000000010')
std_array_p[14].append('000000000000001')

std_array_p.append(Codewords_w)

#Generate a complete standard array
for i in range(len(Codewords_w)):
    temp_p1 = int(Codewords_w[i], 2)^int('100000000000000',2)
    std_array_p[0][i] = str(bin(temp_p1)[2:].zfill(15)) 
    temp_p2 = int(Codewords_w[i], 2)^int('010000000000000',2)
    std_array_p[1][i] = str(bin(temp_p2)[2:].zfill(15)) 
    temp_p3 = int(Codewords_w[i], 2)^int('001000000000000',2)
    std_array_p[2][i] = str(bin(temp_p3)[2:].zfill(15)) 
    temp_p4 = int(Codewords_w[i], 2)^int('000100000000000',2)
    std_array_p[3][i] = str(bin(temp_p4)[2:].zfill(15)) 
    temp_p5 = int(Codewords_w[i], 2)^int('000010000000000',2)
    std_array_p[4][i] = str(bin(temp_p5)[2:].zfill(15)) 
    temp_p6 = int(Codewords_w[i], 2)^int('000001000000000',2)
    std_array_p[5][i] = str(bin(temp_p6)[2:].zfill(15)) 
    temp_p7 = int(Codewords_w[i], 2)^int('000000100000000',2)
    std_array_p[6][i] = str(bin(temp_p7)[2:].zfill(15)) 
    temp_p8 = int(Codewords_w[i], 2)^int('000000010000000',2)
    std_array_p[7][i] = str(bin(temp_p8)[2:].zfill(15)) 
    temp_p9 = int(Codewords_w[i], 2)^int('000000001000000',2)
    std_array_p[8][i] = str(bin(temp_p9)[2:].zfill(15)) 
    temp_p10 = int(Codewords_w[i], 2)^int('000000000100000',2)
    std_array_p[9][i] = str(bin(temp_p10)[2:].zfill(15)) 
    temp_p11 = int(Codewords_w[i], 2)^int('000000000010000',2)
    std_array_p[10][i] = str(bin(temp_p11)[2:].zfill(15)) 
    temp_p12 = int(Codewords_w[i], 2)^int('000000000001000',2)
    std_array_p[11][i] = str(bin(temp_p12)[2:].zfill(15)) 
    temp_p13 = int(Codewords_w[i], 2)^int('000000000000100',2)
    std_array_p[12][i] = str(bin(temp_p13)[2:].zfill(15)) 
    temp_p14 = int(Codewords_w[i], 2)^int('000000000000010',2)
    std_array_p[13][i] = str(bin(temp_p14)[2:].zfill(15)) 
    temp_p15 = int(Codewords_w[i], 2)^int('000000000000001',2)
    std_array_p[14][i] = str(bin(temp_p15)[2:].zfill(15)) 
    
with open("std_array_p_complete.json", "w") as write_file:
    json.dump(std_array_p, write_file)
    
import random
from pprint import pprint
import itertools
from itertools import chain
import numpy as np
import math
import matplotlib.pyplot as plt

font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)

import json

path = 'Data_Files/'

#Codewords for BCH[15,11,1]
with open(path + "Codewords_w.json") as json_file:
    Codewords_w = json.load(json_file)

#Standard Array for BCH[15,11,1]
with open(path + "std_array_p_complete.json") as json_file:
    std_array_p = json.load(json_file)
      
TempRange = np.arange(-25,71,5)
#Uniformity from corrected responses using proposed methodology
Uniformity = np.load(path + 'Uniformity_post_TL.npy')
minEntropy = np.zeros(len(TempRange))
Cond_minEntropy = np.zeros(len(TempRange))
minEntropy_loss = np.zeros(len(TempRange))

#Fixed Helper Data at Reference Temperature
with open(path + 'GResp_25.json') as json_file:
    bins = json.load(json_file)

#%%
for predTemp in TempRange:
    count = np.where(predTemp==TempRange)[0][0]
    bins_unique = {0: [0] * len(bins['0']),
     1: [0] * len(bins['1']),
     2: [0] * len(bins['2']),
     3: [0] * len(bins['3']),
     4: [0] * len(bins['4']),
     5: [0] * len(bins['5']),
     6: [0] * len(bins['6']),
     7: [0] * len(bins['7']),
     8: [0] * len(bins['8']),
     9: [0] * len(bins['9']),
     10: [0] * len(bins['10']),
     11: [0] * len(bins['11']),
     12: [0] * len(bins['12']),
     13: [0] * len(bins['13']),
     14: [0] * len(bins['14']),
     15: [0] * len(bins['15'])}
    
    ## Prob. of each bin
    
    b_star = [0]*16
    q_j = [0]*16
    
    ##	Probability of each Bin	##
    
    prob_p = Uniformity[count]	#Avg. of prob_p list
    b_star = min(prob_p , 1 - prob_p)
    for i in range(16):
    	
    	q_j[i] = pow(b_star, i) * pow( (1-b_star), (15-i) )
        
    print(-1*np.log2(q_j))

    print(" Entropy of x -- ", -1 * np.log2(q_j[0]))
    minEntropy[count] = -1 * np.log2(q_j[0])
   
    count_p_each_bins = [0] * 16
    sum_sp_qj = 0
    no_of_ones = 0
    std_array_p_BCH_unique = np.zeros([16,2048])
    keys = ['15','14','13','12','11','10','9','8','7','6','5','4','3','2','1','0']

    for key in keys:					        ## Iterating over all x's 
        print("key -> ", key)
        for i in range(len(bins[key])):
            print(key, " -> ", ((bins[key][i] )))
            for j in Codewords_w:				## Iterating over all w's
                temp_p = int(bins[key][i], 2)^int(j,2)	        ## Temporary helper data p = x xor w
                
                temp_p_use = str( bin(temp_p)[2:].zfill(len(j))) 
                if(no_of_ones<2**15):
                    if (temp_p_use in chain.from_iterable(std_array_p)) == True:  ## Search for the temporary helper data in Standard Array
                
                        posi_in_p = ([(ix,iy) for ix, row in enumerate(std_array_p) for iy, k in enumerate(row) if k == temp_p_use])		# gives the position of p in std array
                        std_array_p_BCH_unique[posi_in_p[0][0]][posi_in_p[0][1]] = std_array_p_BCH_unique[posi_in_p[0][0]][posi_in_p[0][1]]+ 1 	# posi_in_p[0][0] - row index;; posi_in_p[0][1] - col. index
                        if(std_array_p_BCH_unique[posi_in_p[0][0]][posi_in_p[0][1]]==1):
                            no_of_ones = no_of_ones +1				
                            count_p_each_bins[int(key)] = count_p_each_bins[int(key)] +1 
                else:
                    break
                        
        sum_sp_qj = sum_sp_qj +  count_p_each_bins[int(key)] * q_j[15-int(key)]	
    
    print(" sum_sp_qj - ", sum_sp_qj)
    
    print( " count_p_each_bins - ", count_p_each_bins)
    
    print(" H(X|P) -- ", -1 * np.log2((1/2048)*sum_sp_qj))
    Cond_minEntropy[count] = -1 * np.log2((1/2048)*sum_sp_qj)
    minEntropy_loss[count] = minEntropy[count] - Cond_minEntropy[count]
    
    print("no_of_ones in std. array - ",no_of_ones)

#%%
np.save(path + 'minEntropy_post_TL.npy',minEntropy)
np.save(path + 'minEntropy_Loss_fixed_p_ECC_TL.npy',minEntropy_loss)

 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 22:29:11 2022

@author: 
"""

#Error Study in 3x3 window
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import warnings
import scipy
from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer
from sklearn.pipeline import make_pipeline

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# This code learns the # of errors in a 2x2 window using splines 
# Re-callibrates the golden response and applies ECC to correct the remaining error

font = {'family': 'arial',
        'color':  'black',
        'weight': 'ultralight',
        'size': 16,
        }

path = ''
GoldenRespAmbient = np.load(path+'GResp_volt_Zero_ambient_Oct11.npy')
GoldenResp = np.load(path+'GResp_volt_Zero_all_Oct11.npy')
Reliability_PUF = np.load(path+'Rel_volt_Zero_ambient_Oct11.npy')
CorrectedResp = np.zeros(np.shape(GoldenResp))

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
#%%

#MR = (np.logical_or(Rel_25 == 8,np.logical_or(Rel_25 == 9,np.logical_or(Rel_25 == 10,Rel_25 == 11)))).astype(int)

#Bitpatterns (combinatorics part)
#The initial response is all zeroes
bitpattern_list = np.zeros([512,10])
exhaustive_list = np.zeros([512,5])
exhaustive_cont = np.zeros([512])
exhaustive_cont_noresp = np.zeros([512,2])

for i in range(512):
    instance = np.zeros(9)
    temp = np.binary_repr(i, width=9)
    for j in range(9):
        instance[j] = int(temp[j])
    bitpattern_list[i,0:9] = instance
    instance = instance.reshape([3,3])
    Win1 = instance[0:2,0:2]
    Win2 = instance[0:2,1:3]
    Win3 = instance[1:3,0:2]
    Win4 = instance[1:3,1:3]
    exhaustive_list[i,0] = np.sum(Win1)
    exhaustive_list[i,1] = np.sum(Win2)
    exhaustive_list[i,2] = np.sum(Win3)
    exhaustive_list[i,3] = np.sum(Win4)
    exhaustive_list[i,4] = instance[1,1]
    exhaustive_cont[i] = 10000*np.sum(Win1)+1000*np.sum(Win2)+100*np.sum(Win3)+ 10*np.sum(Win4) + instance[1,1]
    exhaustive_cont_noresp[i,0] = (10000*np.sum(Win1)+1000*np.sum(Win2)+100*np.sum(Win3)+ 10*np.sum(Win4))/10
    bitpattern_list[i,9] = exhaustive_cont_noresp[i,0] 
    exhaustive_cont_noresp[i,1] = instance[1,1]
    
#unique is the array of all possible # of errors in the neighbourhood of the middel cell in 3x3 matrix
unique, count = np.unique(exhaustive_cont_noresp[:,0], return_counts=True)
prediction = np.zeros(len(unique))
correction = np.zeros(len(unique))
for i in range(len(unique)):
    ind = np.where(exhaustive_cont_noresp[:,0] == unique[i])[0]
    prediction[i] = np.sum(exhaustive_cont_noresp[ind,1])/count[i]
    correction[i] = prediction[i]
    if(prediction[i]<=0.5):
        prediction[i] = 1- prediction[i]
            
#Resp at Temp T
Volt_all = np.array([7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,
                     10,10.2,10.4,10.6,10.8,11,11.2,11.4,11.6,11.8,12])
refloc = np.where(Volt_all==7)[0][0]
BoardTotal = 5
BoardRange = np.arange(BoardTotal)
rng = 2
nChal = 6815
respW = 32
R_overlap = nChal - rng + 1
C_overlap = respW - rng + 1

Window_errors_all = np.load(path + 'Window_error_overlapping_2x2_volt_Zero_Oct11.npy')
Dataset = Window_errors_all.reshape([BoardTotal,len(Volt_all),R_overlap*C_overlap])
Window_Acc = np.zeros([BoardTotal,len(Volt_all)])
MLECC_BER = np.zeros([BoardTotal,len(Volt_all)])
# Machine Learning and Golden Resp Update Part
Set = [7.4]
BSet = [1]
#for predVolt in Volt_all:
for predVolt in Set:
    print(predVolt)  
    vidx = np.where(Volt_all==predVolt)[0][0]
    #for board in range(BoardTotal):      
    for board in BSet:      
        flip_probT0 = np.zeros(np.shape(Reliability_PUF[board,:,:]))
                
        for i in range(nChal):
            for j in range(respW):
                flip_probT0[i,j] = Reliability_PUF[board,i,j]

        MR = np.where(flip_probT0>90)
               
        #The reference file to compute the errors
        
        Golden_response = GoldenResp[board,refloc,:,:]
        #Resp at temp - predTemp
        Resp_erroneous = GoldenResp[board,vidx,:,:]
        
        Resp_corrected = copy.deepcopy(Resp_erroneous)
        
        #Predict Window errors at temp T using spline fit
        Window_errors_overlap = np.zeros([R_overlap,C_overlap])
        
        #B-spline fit only with Temperature Data
        Pred  = np.zeros([R_overlap*C_overlap])
        
        
        #Forward
        if(predVolt==7):
            X = np.array([[7]])
            trainInd = 0
            testInd = 0
        elif(predVolt==7.2):
            X = np.array([[7]])
            trainInd = 0
            testInd = 1
        elif(predVolt==7.4):
            X = np.array([[7],[7.2]])
            trainInd = np.arange(2)
            testInd = 2
        elif(predVolt==7.6):
            X = np.array([[7],[7.2],[7.4]])
            trainInd = np.arange(3)
            testInd = 3
        elif(predVolt==7.8):
            X = np.array([[7],[7.2],[7.4],[7.6]])
            trainInd = np.arange(4)
            testInd = 4
        elif(predVolt==8):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8]])
            trainInd = np.arange(5)
            testInd = 5
        elif(predVolt==8.2):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8]])
            trainInd = np.arange(6)
            testInd = 6
        elif(predVolt==8.4):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2]])
            trainInd = np.arange(7)
            testInd = 7
        elif(predVolt==8.6):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4]])
            trainInd = np.arange(8)
            testInd = 8
        elif(predVolt==8.8):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6]])
            trainInd = np.arange(9)
            testInd = 9
        elif(predVolt==9):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8]])
            trainInd = np.arange(10)
            testInd = 10
        elif(predVolt==9.2):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9]])
            trainInd = np.arange(11)
            testInd = 11
        elif(predVolt==9.4):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2]])
            trainInd = np.arange(12)
            testInd = 12
        elif(predVolt==9.6):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4]])
            trainInd = np.arange(13)
            testInd = 13
        elif(predVolt==9.8):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6]])
            trainInd = np.arange(14)
            testInd = 14
        elif(predVolt==10):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8]])
            trainInd = np.arange(15)
            testInd = 15
        elif(predVolt==10.2):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10]])
            trainInd = np.arange(16)
            testInd = 16
        elif(predVolt==10.4):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2]])
            trainInd = np.arange(17)
            testInd = 17
        elif(predVolt==10.6):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4]])
            trainInd = np.arange(18)
            testInd = 18
        elif(predVolt==10.8):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6]])
            trainInd = np.arange(19)
            testInd = 19
        elif(predVolt==11):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8]])
            trainInd = np.arange(20)
            testInd = 20
        elif(predVolt==11.2):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8],[11]])
            trainInd = np.arange(21)
            testInd = 21
        elif(predVolt==11.4):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8],[11],[11.2]])
            trainInd = np.arange(22)
            testInd = 22
        elif(predVolt==11.6):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8],[11],[11.2],[11.4]])
            trainInd = np.arange(23)
            testInd = 23
        elif(predVolt==11.8):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8],[11],[11.2],[11.4],[11.6]])
            trainInd = np.arange(24)
            testInd = 24
        elif(predVolt==12):
            X = np.array([[7],[7.2],[7.4],[7.6],[7.8],[8],[8.2],[8.4],[8.6],[8.8],[9],[9.2],[9.4],[9.6],[9.8],[10],[10.2],[10.4],[10.6],[10.8],[11],[11.2],[11.4],[11.6],[11.8]])
            trainInd = np.arange(25)
            testInd = 25
            
        if(predVolt != 7 and predVolt != 7.2):
            #for i in range(8000,8001):
            for i in range(R_overlap*C_overlap):
 
                learning_param = np.zeros(len(trainInd))
                learning_param = Dataset[board,trainInd,i]

                model = make_pipeline(SplineTransformer(n_knots=10, degree=4), Ridge(alpha=1e-3))
                X_train = X
                model.fit(X_train,learning_param)
                X_test = np.array([predVolt])[:,np.newaxis]
                
                y_pred = model.predict(X_test)
                #y_pred = poly(np.array([predTemp]).reshape(1,-1))
                #Arbiter In the End
                if(y_pred<0.5):
                    y_pred = 0
                elif(y_pred>=0.5 and y_pred<1.5):
                    y_pred = 1
                elif(y_pred>=1.5 and y_pred<2.5):
                    y_pred = 2
                elif(y_pred>=2.5 and y_pred<3.5):
                    y_pred = 3
                elif(y_pred>=3.5):
                    y_pred = 4
                    
                Pred[i] = y_pred
            
            #if(np.sum(Dataset[board,testInd,:]==Pred)<np.sum(Dataset[board,testInd,:]==Dataset[board,0,:])):
                #Pred = Dataset[board,0,:]  
            print(np.unique(Pred,return_counts=True),np.unique(Dataset[board,testInd,:],return_counts=True))
            wAcc = (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Pred))) 
            print("Absolute Equality Accurracy: %.6f" % (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Pred))))
        else:
            Pred = Dataset[board,trainInd,:]
            wAcc = (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Dataset[board,testInd,:])))
            print("Absolute Equality Accurracy: %.6f" % (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Dataset[board,testInd,:]))))
        Window_Acc[board,vidx] =  wAcc        
        Window_errors_overlap = Pred.reshape([R_overlap,C_overlap])
        Window_error_inside = np.zeros([int(nChal/rng),int(respW/rng)])
        for i in range(rng-1,R_overlap,rng):
            for j in range(rng-1,C_overlap,rng):
                Window_error_inside[int(i/rng),int(j/rng)] = Window_errors_overlap[i,j]
        
        #Correction of errors using ML and coverage
        for i in range(rng-1,R_overlap-1):
            for j in range(rng-1,C_overlap-1):
                if(Window_error_inside[int(np.ceil(i/rng))-1,int(np.ceil(j/rng))-1]>1):
                    Window_code = (10000*Window_errors_overlap[i-1,j-1]+1000*Window_errors_overlap[i-1,j]
                    +100*Window_errors_overlap[i,j-1]+ 10*Window_errors_overlap[i,j])/10
                    if(len(np.where(unique==Window_code)[0])!=0):
                        idx = np.where(unique==Window_code)[0]
                        if(prediction[idx]==1):
                            if(correction[idx]==1):
                                #100% certainity that bit flip happened and need that position in the Golden response to 
                                #update that position
                                Golden_response[i,j] = 1 - Golden_response[i,j]
                                Window_error_inside[int(np.ceil(i/rng))-1,int(np.ceil(j/rng))-1] = Window_error_inside[int(np.ceil(i/rng))-1,int(np.ceil(j/rng))-1] -1
                        else:
                            #<100% certainity that bit flip happened and need to use the reliability information
                            #Using reliability information to update that poistion in the golden response
                            
                            error_bit_pattern = bitpattern_list[np.where(bitpattern_list[:,9]==Window_code)[0],0:9]
                            #print(Window_code)
                            most_probable_pattern_prob = np.zeros(len(error_bit_pattern[:,0]))
                            #print(error_bit_pattern)
                            Rel_unfolded = flip_probT0[i-1:i+rng,j-1:j+rng].reshape(9)
                            #print(Rel_unfolded)
                            for k in range(len(error_bit_pattern[:,0])):
                                most_probable_pattern_prob[k] = 1
                                for l in range(9):
                                    if(error_bit_pattern[k,l]==0):
                                        #Case when the bit-flip does not happen and prob of that
                                        most_probable_pattern_prob[k] = most_probable_pattern_prob[k]*Rel_unfolded[l]
                                    else:
                                        #Case when the bit-flip does happen and prob of that
                                        most_probable_pattern_prob[k] = most_probable_pattern_prob[k]*(1-Rel_unfolded[l])
                            #print(most_probable_pattern_prob)            
                            most_probable_pattern = error_bit_pattern[np.argmax(most_probable_pattern_prob),0:9]
                            #print(most_probable_pattern)
                            if(most_probable_pattern[4]==1):
                                #Flip when the most probable pattern has centre bit as one
                                Golden_response[i,j] = 1 - Golden_response[i,j]
                                Window_error_inside[int(np.ceil(i/rng))-1,int(np.ceil(j/rng))-1] = Window_error_inside[int(np.ceil(i/rng))-1,int(np.ceil(j/rng))-1] -1
        for i in range(rng-1,R_overlap,rng):
            for j in range(rng-1,R_overlap,rng):
                if(np.count_nonzero(Resp_corrected[i:i+rng,j:j+rng]!=Golden_response[i:i+rng,j:j+rng])==1):
                    Resp_corrected[i:i+rng,j:j+rng] =  Golden_response[i:i+rng,j:j+rng]
                    
        CorrectedResp[board,vidx,:,:] = Resp_corrected      
        
        #Calculating performace of error scheme leavng out the unrelaible bits
        Acc3 = 1-(Resp_corrected[MR]==Golden_response[MR]).sum()/(np.shape(MR)[1])
        print(predVolt,Acc3*100)
        MLECC_BER[board,vidx] = Acc3
        
np.save(path+ 'Corrected_Resp_Zero_Volt_ContinualLearning_Oct11.npy',CorrectedResp) 

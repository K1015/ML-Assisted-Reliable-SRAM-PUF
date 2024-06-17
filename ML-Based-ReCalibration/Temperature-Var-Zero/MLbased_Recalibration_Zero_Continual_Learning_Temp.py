 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 22:29:11 2022

@author: saionroy
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
GoldenRespAmbient = np.load(path+'GResp_temp_Zero_ambient_Oct14.npy')
GoldenResp = np.load(path+'GResp_temp_Zero_all_Oct14.npy')
Reliability_PUF = np.load(path+'Rel_temp_Zero_ambient_Oct14.npy')
CorrectedResp = np.zeros(np.shape(GoldenResp))

Temp_all = np.append([-25],np.arange(-23.5,70,2.5))
refloc = np.where(Temp_all==24)[0][0]

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
TempRange = np.append([-25],np.arange(-23.5,70,2.5))
BoardTotal = 10
BoardRange = np.arange(BoardTotal)
testBoard = [3]
rng = 2
nChal = 6815
respW = 32
R_overlap = nChal - rng + 1
C_overlap = respW - rng + 1

Window_errors_all = np.load(path + 'Window_error_overlapping_Zero_2x2_Temp_Oct14.npy')
Pred_Data = np.zeros([BoardTotal,len(TempRange),R_overlap,C_overlap])
Dataset = Window_errors_all.reshape([BoardTotal,len(TempRange),R_overlap*C_overlap])
Window_Acc = np.zeros([BoardTotal,len(Temp_all)])
MLECC_BER = np.zeros([BoardTotal,len(Temp_all)])
# Machine Learning and Golden Resp Update Part
Set = [69]
count = 0
for predTemp in Set:
    count = 0
    ct = 0
    cn = 0
    print(predTemp)  
    tidx = np.where(TempRange==predTemp)[0][0]
    for board in testBoard:      
        flip_probT0 = np.zeros(np.shape(Reliability_PUF[board,:,:]))
                
        for i in range(nChal):
            for j in range(respW):
                flip_probT0[i,j] = Reliability_PUF[board,i,j]

        MR = np.where(flip_probT0>90)
               
        #The reference file to compute the errors
        
        Golden_response = GoldenResp[board,refloc,:,:]
        #Resp at temp - predTemp
        Resp_erroneous = GoldenResp[board,tidx,:,:]
        
        Resp_corrected = copy.deepcopy(Resp_erroneous)
        
        #Predict Window errors at temp T using spline fit
        Window_errors_overlap = np.zeros([R_overlap,C_overlap])
        
        #B-spline fit only with Temperature Data
        Pred  = np.zeros([R_overlap*C_overlap])
        
        #Forward
        if(predTemp==69):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54],[56.5],[59],[61.5],[64],[66.5]])
            trainInd = np.arange(20,38)
            testInd = 38
        elif(predTemp==66.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54],[56.5],[59],[61.5],[64]])
            trainInd = np.arange(20,37)
            testInd = 37
        elif(predTemp==64):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54],[56.5],[59],[61.5]])
            trainInd = np.arange(20,36)
            testInd = 36
        elif(predTemp==61.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54],[56.5],[59]])
            trainInd = np.arange(20,35)
            testInd = 35
        elif(predTemp==59):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54],[56.5]])
            trainInd = np.arange(20,34)
            testInd = 34
        elif(predTemp==56.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5],[54]])
            trainInd = np.arange(20,33)
            testInd = 33
        elif(predTemp==54):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49],[51.5]])
            trainInd = np.arange(20,32)
            testInd = 32
        elif(predTemp==51.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5],[49]])
            trainInd = np.arange(20,31)
            testInd = 31
        elif(predTemp==49):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44],[46.5]])
            trainInd = np.arange(20,30)
            testInd = 30
        elif(predTemp==46.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5],
                          [44]])
            trainInd = np.arange(20,29)
            testInd = 29
        elif(predTemp==44):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39],[41.5]])
            trainInd = np.arange(20,28)
            testInd = 28
        elif(predTemp==41.5):
            X = np.array([[24],[26.5],[29],[31.5],[34],[36.5],[39]])
            trainInd = np.arange(20,27)
            testInd = 27
        elif(predTemp==39):
           X = np.array([[24],[26.5],[29],[31.5],[34],[36.5]])
           trainInd = np.arange(20,26)
           testInd = 26
        elif(predTemp==36.5):
           X = np.array([[24],[26.5],[29],[31.5],[34]])
           trainInd = np.arange(20,25)
           testInd = 25
        elif(predTemp==34):
           X = np.array([[24],[26.5],[29],[31.5]])
           trainInd = np.arange(20,24)
           testInd = 24
        elif(predTemp==31.5):
           X = np.array([[24],[26.5],[29]])
           trainInd = np.arange(20,23)
           testInd = 23
        elif(predTemp==29):
           X = np.array([[24],[26.5]])
           trainInd = np.arange(20,22)
           testInd = 22
        elif(predTemp==26.5):
            X = np.array([[24]])
            trainInd = 20
            testInd = 21
        elif(predTemp==24):
            X = np.array([[24]])
            trainInd = 20
            testInd = 20
        elif(predTemp==21.5):
            X = np.array([[24]])
            trainInd = 20
            testInd = 19
        elif(predTemp==19):
            X = np.array([[21.5],[24]])
            trainInd = np.arange(19,21)
            testInd = 18
        elif(predTemp==16.5):
            X = np.array([[19],
                          [21.5],[24]])
            trainInd = np.arange(18,21)
            testInd = 17
        elif(predTemp==14):
            X = np.array([[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(17,21)
            testInd = 16
        elif(predTemp==11.5):
            X = np.array([[14],[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(16,21)
            testInd = 15
        elif(predTemp==9):
            X = np.array([[11.5],[14],[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(15,21)
            testInd = 14
        elif(predTemp==6.5):
            X = np.array([[9],[11.5],[14],[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(14,21)
            testInd = 13
        elif(predTemp==4):
            X = np.array([[6.5],[9],[11.5],[14],[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(13,21)
            testInd = 12
        elif(predTemp==1.5):
            X = np.array([[4],[6.5],[9],[11.5],[14],[16.5],[19],
                          [21.5],[24]])
            trainInd = np.arange(12,21)
            testInd = 11
        elif(predTemp==-1):
            X = np.array([[1.5],[4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(11,21)
            testInd = 10
        elif(predTemp==-3.5):
            X = np.array([[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(10,21)
            testInd = 9
        elif(predTemp==-6):
            X = np.array([[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(9,21)
            testInd = 8
        elif(predTemp==-8.5):
            X = np.array([[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(8,21)
            testInd = 7
        elif(predTemp==-11):
            X = np.array([[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(7,21)
            testInd = 6
        elif(predTemp==-13.5):
            X = np.array([[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(6,21)
            testInd = 5
        elif(predTemp==-16):
            X = np.array([[-13.5],[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(5,21)
            testInd = 4
        elif(predTemp==-18.5):
            X = np.array([[-16],[-13.5],[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(4,21)
            testInd = 3
        elif(predTemp==-21):
            X = np.array([[-18.5],[-16],[-13.5],[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(3,21)
            testInd = 2
        elif(predTemp==-23.5):
            X = np.array([[-21],[-18.5],[-16],[-13.5],[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(2,21)
            testInd = 1
        elif(predTemp==-25):
            X = np.array([[-23.5],[-21],[-18.5],[-16],[-13.5],[-11],[-8.5],[-6],[-3.5],[-1],[1.5],
                          [4],[6.5],[9],[11.5],[14],[16.5],[19],[21.5],[24]])
            trainInd = np.arange(1,21)
            testInd = 0
            
        if(predTemp != 24 and predTemp != 26.5 and predTemp != 21.5):
            #for i in range(8000,8001):
            for i in range(R_overlap*C_overlap):
                transfer_board = board
                
                learning_param = np.zeros(len(trainInd))
                learning_param = Dataset[transfer_board,trainInd,i]
                model = make_pipeline(SplineTransformer(n_knots=10, degree=4), Ridge(alpha=1e-3))
                X_train = X
                model.fit(X_train,learning_param)
                X_test = np.array([predTemp])[:,np.newaxis]
                
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
                
            wAcc = (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Pred)))       
            print("Absolute Equality Accurracy: %.6f" % (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Pred))))
        else:
            Pred = Dataset[board,trainInd,:]
            wAcc = (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Dataset[board,testInd,:])))
            print("Absolute Equality Accurracy: %.6f" % (100*(np.sum(Dataset[board,testInd,:]==Pred)/len(Dataset[board,testInd,:]))))
        Window_Acc[board,tidx] =  wAcc       
        Window_errors_overlap = Pred.reshape([R_overlap,C_overlap])
        Pred_Data[board,tidx,:,:] = Window_errors_overlap 
        
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
                            cn = cn + 1
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
                    
        CorrectedResp[board,tidx,:,:] = Resp_corrected
        
        #Calculating performace of error scheme leaving our the corner cells
        Acc1 = 1-(Resp_corrected[1:127,1:63]==Golden_response[1:127,1:63]).sum()/(126*62)
        #print(predTemp,Acc1*100)
        Acc2 = 1-(Resp_corrected==Golden_response).sum()/(126*62)
        #print(Acc2*100)
        
        #Calculating performace of error scheme leavng out the unrelaible bits
        Acc3 = 1-(Resp_corrected[MR]==Golden_response[MR]).sum()/(np.shape(MR)[1])
        print(predTemp,Acc3*100)
        MLECC_BER[board,tidx] = Acc3
        
np.save(path + 'CorrectedResp_Zero_Temp_ContinualLearning.npy',CorrectedResp) 
np.save(path + 'PredWErrors_Zero_Temp_ContinualLearning.npy',Pred_Data) 
        
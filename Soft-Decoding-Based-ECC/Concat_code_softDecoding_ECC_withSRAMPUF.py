
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
from scipy import stats
from reedmuller import reedmuller

# This code corrects the errors in PUF using a soft decoder based implementation of Conatenated Code
# We use Repetition (3,1,3) and Reed-Muller (2,4) as an example 
# Uses ML methods to partially correct for errors before ECC

font = {'family': 'arial',
        'color':  'black',
        'weight': 'bold',
        'size': 16,
        }

path = ''

# All the data files are for Arduino UNO Board Number 1

GoldenResp = np.load(path+'GResp_temp_all_Br1.npy')
CorrectedResp_TransferL = np.load(path+'TransferL_corrected_Resp_temp_all_Br1.npy')
CorrectedResp_ContinualL = np.load(path+'ContinualL_corrected_Resp_temp_all_Br1.npy')
Reliability_PUF = np.load(path+'Reliability_PUF_ambient_Br1.npy')

TempRange = np.arange(-23.5,70,2.5)
# Reference temperature is 24C

refloc = np.where(TempRange==24)[0][0]
#%%
#This code is for 1 Board only

BoardTotal = 1
boardID = 0
nrow = 128
ncol = 64

BoardRange = np.arange(BoardTotal)

GoldenResp = GoldenResp.reshape([len(BoardRange),len(TempRange),nrow,ncol])
CorrectedResp_TransferL = CorrectedResp_TransferL.reshape([len(BoardRange),len(TempRange),nrow,ncol])
CorrectedResp_ContinualL  = CorrectedResp_ContinualL.reshape([len(BoardRange),len(TempRange),nrow,ncol])
Reliability_PUF = Reliability_PUF.reshape([len(BoardRange),nrow,ncol])

BER_Accuracy_Vanilla = np.zeros([len(BoardRange),len(TempRange)])
BER_Accuracy_PostECC = np.zeros([len(BoardRange),len(TempRange)])
BER_Accuracy_PostML_ECC = np.zeros([len(BoardRange),len(TempRange)])
BER_Accuracy_PostML_ECC_sp = np.zeros([len(BoardRange),len(TempRange)])

#%%
#Resp at Temp T
#C1 is RM(2,4); r = 2, m = 4
#C2 is Rep(3,1,3)

m = 4
r = 2

rm = reedmuller.ReedMuller(r, m)

n1 = 2**m #2**m
n2 = 3
CodeWidth = n1*n2 #n1*n2

k1 = 11
k2 = 1
MessageWidth = k1*k2 #k1*k2
message_list = np.zeros([2**k1,k1])
for i in range(2**k1):
    for j in range(k1):
        message_list[i,j] = int(np.binary_repr(i,width=k1)[j])

codeword_decoding_mat = np.zeros([2**k1,n1])
for i in range(2**k1):
    codeword_decoding_mat[i,:] = np.array(rm.encode(message_list[i,:]))

codeword_ref_mat = copy.deepcopy(codeword_decoding_mat)
codeword_decoding_mat[codeword_decoding_mat==1] = -1
codeword_decoding_mat[codeword_decoding_mat==0] = 1

#%%        
d1 = 2**(m-r)
d2 = 3

t1 = int(np.floor((d1-1)/2))
t2 = int(np.floor((d2-1)/2))

l1 = int(np.ceil(171/k1))
lu = l1*n1
l2 = int(np.ceil(lu/k2))

PUFWidth = int(l1*n1*n2)
Reliability_PUF = Reliability_PUF/100
error_prob_PUF = 1-Reliability_PUF
soft_info = np.array([np.floor(math.log(1-x+1e-30,1.8) - math.log(x+1e-30,1.8)) for x in error_prob_PUF.reshape(-1)]).reshape(np.shape(error_prob_PUF))

#%%
def F(x,y):
    return np.sign(x*y)*np.min([np.abs(x),np.abs(y)])

def G(s,x,y):
    return np.floor(0.5*(np.sign(s)*x+y))

def SDML_dec(n,L):
    L_star = np.sum(L)
    return np.ones(n)*L_star

def GMC_dec(r,m,L):
    if(r==0):
        L_star = SDML_dec(2**m, L)
        #print('L_star(r=0)',L_star.shape)
    elif(r==m):
        L_star = L
        #print('L_star(r=m)',L_star.shape)
    else:
        L_sup1 = np.zeros(2**(m-1))
        for j in range(2**(m-1)):
            L_sup1[j] = F(L[2*j],L[2*j+1])
        L_sup1star = GMC_dec(r-1,m-1,L_sup1)
        #print('L_sup1*',L_sup1star.shape)
        L_sup2 = np.zeros(2**(m-1))
        for j in range(2**(m-1)):
            L_sup2[j] = G(L_sup1[j],L[2*j],L[2*j+1])
        L_sup2star = GMC_dec(r,m-1,L_sup2)
        #print('L_sup2*',L_sup2star.shape)
        L_star = np.zeros(2**m)
        for j in range(2**m):
            if(np.mod(j,2)==0):
                #print('here is error - (j, L_sup1*, L_sup2*)',j,L_sup1star.shape,L_sup2star.shape)
                L_star[j] = F(L_sup1star[int(j/2)],L_sup2star[int(j/2)])
            else:
                L_star[j] = L_sup2star[int(j/2)]
    return L_star

#%%
for board in range(BoardTotal):
    MR = np.where(Reliability_PUF[board,:,:]>0.9)
    Golden_response =  GoldenResp[board,refloc,MR[0],MR[1]].reshape(-1)
    PUFDim = int(np.floor(Golden_response.shape[0]/PUFWidth))
    
    Raw_response = Golden_response[:PUFDim*PUFWidth].reshape([PUFDim,PUFWidth])
    Helper_data = np.zeros([PUFDim,PUFWidth])
    codeword_l1_all = np.zeros([PUFDim,l1,n1])
    codeword_all = np.zeros([PUFDim, PUFWidth])
    
    for i in range(PUFDim):
        #Generate l1 Random Codewords
        message = np.zeros([l1,k1])
        codewordsl1 = np.zeros([l1,n1])
        for lind in range(l1):
            message[lind,:] = np.random.randint(2,size=MessageWidth)
            codewordsl1[lind,:] = np.array(rm.encode(message[lind,:]))
        codeword_l1_all[i,:,:] = codewordsl1
        codewordsl1_rev = codewordsl1
        #concatenate codeword for C1
        codewordsl1_flat = codewordsl1_rev.reshape(-1)
        
        messagel2 = codewordsl1_flat.reshape([l2,k2])
        codewordsl2 = np.repeat(messagel2,n2,axis=1)
        #concatenate codeword for C2 (no reversing required)
        codewordsl2_flat = codewordsl2.reshape(-1)
        codeword_all[i,:] = codewordsl2_flat
        Helper_data[i,:] = np.bitwise_xor(Raw_response[i,:].astype(int),codewordsl2_flat.astype(int))
        
    for predTemp in TempRange:  
        print(predTemp)    
        tidx = np.where(TempRange==predTemp)[0][0]
        #tidx = refloc
        
        Resp_Vanilla = GoldenResp[board,tidx,MR[0],MR[1]].reshape(-1) 
        Resp_Vanilla = Resp_Vanilla[:PUFDim*PUFWidth].reshape([PUFDim,PUFWidth])
        
        Resp_corrected_ContinualL = CorrectedResp_ContinualL[board,tidx,MR[0],MR[1]].reshape(-1) 
        Resp_corrected_ContinualL = Resp_corrected_ContinualL[:PUFDim*PUFWidth].reshape([PUFDim,PUFWidth])
        
        Resp_corrected_TransferL = CorrectedResp_TransferL[board,tidx,MR[0],MR[1]].reshape(-1)
        Resp_corrected_TransferL = Resp_corrected_TransferL[:PUFDim*PUFWidth].reshape([PUFDim,PUFWidth])
        
        soft_info_decoding = soft_info[board,MR[0],MR[1]].reshape(-1)
        soft_info_decoding = soft_info_decoding[:PUFDim*PUFWidth].reshape([PUFDim,l2,n2])
        
        Codeword_received = np.zeros([PUFDim,PUFWidth])
        Codeword_MLreceived = np.zeros([PUFDim,PUFWidth])
        Codeword_MLreceived_sp = np.zeros([PUFDim,PUFWidth])
        
        for i in range(PUFDim):               
            Codeword_received[i,:] = np.bitwise_xor(Resp_Vanilla[i,:].astype(int),Helper_data[i,:].astype(int))    
            Codeword_MLreceived[i,:] = np.bitwise_xor(Resp_corrected_ContinualL[i,:].astype(int),Helper_data[i,:].astype(int))
            Codeword_MLreceived_sp[i,:] = np.bitwise_xor(Resp_corrected_TransferL[i,:].astype(int),Helper_data[i,:].astype(int))
        
        Codeword_received[Codeword_received==1] = -1
        Codeword_received[Codeword_received==0] = 1
        Codeword_MLreceived[Codeword_MLreceived==1] = -1
        Codeword_MLreceived[Codeword_MLreceived==0] = 1
        Codeword_MLreceived_sp[Codeword_MLreceived_sp==1] = -1
        Codeword_MLreceived_sp[Codeword_MLreceived_sp==0] = 1
        
        FinalResponse = np.zeros([PUFDim,PUFWidth])
        FinalResponseML = np.zeros([PUFDim,PUFWidth])
        FinalResponseML_sp = np.zeros([PUFDim,PUFWidth])
        
        #Soft-Decoding Step 1 for codeword C2
        #Computing Likelihood
        Message_received = np.sum(np.multiply(Codeword_received.reshape([PUFDim,l2,n2]),soft_info_decoding),axis=2).reshape([PUFDim,l1,n1])
        Message_MLreceived = np.sum(np.multiply(Codeword_MLreceived.reshape([PUFDim,l2,n2]),soft_info_decoding),axis=2).reshape([PUFDim,l1,n1])
        Message_MLreceived_sp = np.sum(np.multiply(Codeword_MLreceived_sp.reshape([PUFDim,l2,n2]),soft_info_decoding),axis=2).reshape([PUFDim,l1,n1])
        
        #Soft-Decoding Step 2 for codeword C1
        #Computing Likelihood
        Codeword_correctedl1_ECC = np.zeros([PUFDim,l1,n1])
        Codeword_MLcorrectedl1_ECC = np.zeros([PUFDim,l1,n1])
        Codeword_MLcorrectedl1_ECC_sp = np.zeros([PUFDim,l1,n1])
        
        for i in range(PUFDim):
            for lind in range(l1):

                Codeword_correctedl1_ECC[i,lind,:] = codeword_ref_mat[np.argmax(np.matmul(codeword_decoding_mat,GMC_dec(r,m,Message_received[i,lind,:]))),:]
                    
                Codeword_MLcorrectedl1_ECC[i,lind,:] = codeword_ref_mat[np.argmax(np.matmul(codeword_decoding_mat,GMC_dec(r,m,Message_MLreceived[i,lind,:]))),:]
                    
                Codeword_MLcorrectedl1_ECC_sp[i,lind,:] = codeword_ref_mat[np.argmax(np.matmul(codeword_decoding_mat,GMC_dec(r,m,Message_MLreceived_sp[i,lind,:]))),:]
                
            codewordsl1_cor_rev = Codeword_correctedl1_ECC[i,:,:]
            
            #concatenate codeword for C1
            codewordsl1_cor_flat = codewordsl1_cor_rev.reshape(-1)
            messagel2_cor = codewordsl1_cor_flat.reshape([l2,k2])
            codewordsl2_cor = np.repeat(messagel2_cor,n2,axis=1)
            #concatenate codeword for C2 (no reversing required)
            codewordsl2_cor_flat = codewordsl2_cor.reshape(-1)
            FinalResponse[i,:] = np.bitwise_xor(Helper_data[i,:].astype(int),codewordsl2_cor_flat.astype(int))
        
            codewordsl1_MLcor_rev = Codeword_MLcorrectedl1_ECC[i,:,:]

            #concatenate codeword for C1
            codewordsl1_MLcor_flat = codewordsl1_MLcor_rev.reshape(-1)
            messagel2_MLcor = codewordsl1_MLcor_flat.reshape([l2,k2])
            codewordsl2_MLcor = np.repeat(messagel2_MLcor,n2,axis=1)
            #concatenate codeword for C2 (no reversing required)
            codewordsl2_MLcor_flat = codewordsl2_MLcor.reshape(-1)
            FinalResponseML[i,:] = np.bitwise_xor(Helper_data[i,:].astype(int),codewordsl2_MLcor_flat.astype(int))
            
            codewordsl1_MLcor_sp_rev = Codeword_MLcorrectedl1_ECC_sp[i,:,:]
            
            #concatenate codeword for C1
            codewordsl1_MLcor_sp_flat = codewordsl1_MLcor_sp_rev.reshape(-1)
            messagel2_MLcor_sp = codewordsl1_MLcor_sp_flat.reshape([l2,k2])
            codewordsl2_MLcor_sp = np.repeat(messagel2_MLcor_sp,n2,axis=1)
            #concatenate codeword for C2 (no reversing required)
            codewordsl2_MLcor_sp_flat = codewordsl2_MLcor_sp.reshape(-1)
            FinalResponseML_sp[i,:] = np.bitwise_xor(Helper_data[i,:].astype(int),codewordsl2_MLcor_sp_flat.astype(int))
    
        BER_Accuracy_Vanilla[board,tidx] = (np.sum(np.abs(Resp_Vanilla-Raw_response)))/(PUFWidth*PUFDim)*100
        BER_Accuracy_PostECC[board,tidx] = (np.sum(np.abs(FinalResponse-Raw_response)))/(PUFWidth*PUFDim)*100
        BER_Accuracy_PostML_ECC[board,tidx] = (np.sum(np.abs(FinalResponseML-Raw_response)))/(PUFWidth*PUFDim)*100
        BER_Accuracy_PostML_ECC_sp[board,tidx] = (np.sum(np.abs(FinalResponseML_sp-Raw_response)))/(PUFWidth*PUFDim)*100
        
        print(BER_Accuracy_PostECC[board,tidx],BER_Accuracy_PostML_ECC[board,tidx],BER_Accuracy_PostML_ECC_sp[board,tidx])    
       
#%%
#For Printing

fig, ax = plt.subplots(figsize=(8,8))
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
plt.ylabel('BER in %',fontsize=42,fontdict=font)
plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=42,fontdict=font)
ax.plot(TempRange[3::2], BER_Accuracy_Vanilla[boardID,3::2],marker="o",markersize=15,color='blue',label='Raw BER without ECC')
ax.plot(TempRange[3::2], BER_Accuracy_PostECC[boardID,3::2],marker="d",markersize=15,color='red',label=r'Soft Decoding with Concatenated Code$^*$')
ax.plot(TempRange[3::2], BER_Accuracy_PostML_ECC_sp[boardID,3::2],marker="s",markersize=15,color='green',label=r'Soft Decoding$^*$+Continual Learning')
ax.plot(TempRange[3::2], BER_Accuracy_PostML_ECC[boardID,3::2],marker="s",markersize=15,color='darkorange',label=r'Soft Decoding$^*$+Transfer Learning')
ax.text(0.65, 0.8, r'$^*$Repetition(3,1,3)'+'\n'+'+ Reed-Muller(2,4)',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=26,
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round'))
ax.set_xlim([-30,75])
ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
#plt.legend(loc='best', prop={'size': 28, 'family':'Arial'})
plt.legend(frameon=True,framealpha=1)
plt.legend(loc='upper center', ncol=1,bbox_to_anchor=(0.51, 1.52),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Soft_Decoding_BER_SRAM_PUF_Temp_all.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()

#%%

import operator as op
from functools import reduce

def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2

KER_Accuracy_int_Vanilla = np.ones(len(BER_Accuracy_Vanilla[boardID,:]))
KER_Accuracy_int_PostECC = np.ones(len(BER_Accuracy_PostECC[boardID,:]))
KER_Accuracy_int_PostML_ECC = np.ones(len(BER_Accuracy_PostML_ECC[boardID,:]))
KER_Accuracy_int_PostML_ECC_sp = np.ones(len(BER_Accuracy_PostML_ECC_sp[boardID,:]))

for i in range(len(KER_Accuracy_int_Vanilla)):
    for j in range(t2):
        KER_Accuracy_int_Vanilla[i] = KER_Accuracy_int_Vanilla[i] - ncr(n2,j)*(1-BER_Accuracy_Vanilla[boardID,i]/100)**(n2-j)*(BER_Accuracy_Vanilla[boardID,i]/100)**j
        KER_Accuracy_int_PostECC[i] = KER_Accuracy_int_PostECC[i] - ncr(n2,j)*(1-BER_Accuracy_PostECC[boardID,i]/100)**(n2-j)*(BER_Accuracy_PostECC[boardID,i]/100)**j
        KER_Accuracy_int_PostML_ECC[i] = KER_Accuracy_int_PostML_ECC[i] - ncr(n2,j)*(1-BER_Accuracy_PostML_ECC[boardID,i]/100)**(n2-j)*(BER_Accuracy_PostML_ECC[boardID,i]/100)**j
        KER_Accuracy_int_PostML_ECC_sp[i] = KER_Accuracy_int_PostML_ECC_sp[i] - ncr(n2,j)*(1-BER_Accuracy_PostML_ECC_sp[boardID,i]/100)**(n2-j)*(BER_Accuracy_PostML_ECC_sp[boardID,i]/100)**j
        
KER_Accuracy_Vanilla = np.ones(len(BER_Accuracy_Vanilla[boardID,:]))
KER_Accuracy_PostECC = np.ones(len(BER_Accuracy_PostECC[boardID,:]))
KER_Accuracy_PostML_ECC = np.ones(len(BER_Accuracy_PostML_ECC[boardID,:]))
KER_Accuracy_PostML_ECC_sp = np.ones(len(BER_Accuracy_PostML_ECC_sp[boardID,:]))

for i in range(len(KER_Accuracy_Vanilla)):
    for j in range(t1):
        KER_Accuracy_Vanilla[i] = KER_Accuracy_Vanilla[i] - ncr(n1,j)*(1-KER_Accuracy_int_Vanilla[i]/100)**(n1-j)*(KER_Accuracy_int_Vanilla[i]/100)**j
        KER_Accuracy_PostECC[i] = KER_Accuracy_PostECC[i] - ncr(n1,j)*(1-KER_Accuracy_int_PostECC[i]/100)**(n1-j)*(KER_Accuracy_int_PostECC[i]/100)**j
        KER_Accuracy_PostML_ECC[i] = KER_Accuracy_PostML_ECC[i] - ncr(n1,j)*(1-KER_Accuracy_int_PostML_ECC[i]/100)**(n1-j)*(KER_Accuracy_int_PostML_ECC[i]/100)**j
        KER_Accuracy_PostML_ECC_sp[i] = KER_Accuracy_PostML_ECC_sp[i] - ncr(n1,j)*(1-KER_Accuracy_int_PostML_ECC_sp[i]/100)**(n1-j)*(KER_Accuracy_int_PostML_ECC_sp[i]/100)**j

        
#%%
fig, ax = plt.subplots(figsize=(10,9))
plt.yticks(fontname = "Arial",weight='regular') 
plt.xticks(fontname = "Arial",weight='regular') 
plt.yticks(ticks=np.array([0,1e-14,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1,10]))
ax.set_yscale('log')
ax.set_ylim(9.9e-7,10)
plt.ylabel('Key Error Rate',fontsize=42,fontdict=font)
plt.xlabel(r'Temperature in $^{\circ}$C',fontsize=43,fontdict=font)
ax.plot(TempRange[1::2], KER_Accuracy_Vanilla[1::2],marker="o",markersize=15,color='blue',label='Raw BER without ECC')
ax.plot(TempRange[1::2], KER_Accuracy_PostECC[1::2],marker="d",markersize=15,color='red',label=r'Soft Decoding with Concatenated Code$^*$')
ax.plot(TempRange[1::2], KER_Accuracy_PostML_ECC_sp[1::2],marker="s",markersize=15,color='green',label=r'Soft Decoding$^*$+Continual Learning')
ax.plot(TempRange[1::2], KER_Accuracy_PostML_ECC[1::2],marker="s",markersize=15,color='darkorange',label=r'Soft Decoding$^*$+Transfer Learning')

ax.text(0.65, 0.85, r'$^*$Repetition(3,1,3)'+'\n'+'+ Reed-Muller(2,4)',
        verticalalignment='bottom', horizontalalignment='center',
        transform=ax.transAxes,
        color='black', fontsize=26,
        bbox=dict(facecolor='none', edgecolor='none', boxstyle='round'))

ax.xaxis.set_tick_params(labelsize=32)
ax.yaxis.set_tick_params(labelsize=32)
ax.grid(True)
plt.legend(frameon=True,framealpha=1)
plt.legend(loc='upper center', ncol=1,bbox_to_anchor=(0.51, 1.48),prop={'size': 28, 'family':'Arial', 'weight':'regular'}         
           ,edgecolor='black',columnspacing=0.3,handlelength=1.0,handletextpad=0.5)
plt.savefig('Soft_Decoding_KER_SRAM_PUF_Temp_all.pdf', format='pdf', transparent=True,bbox_inches='tight')
plt.show()
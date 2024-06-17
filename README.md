# ML-Assisted-Reliable-SRAM-PUF
Code for the paper titled "Enhancing SRAM-Based PUF Reliability Through Machine Learning-Aided Calibration Techniques". Please note that the codes and data for replicating the plots in the paper will be provided upon the request of the camera-ready version. In the current version, the raw data for the Zero boards are not included in this repository due to space limitations (.csv file size > 100MB) and can be provided upon request. This repository contains the following:

## Raw Data for Arduino UNO
The raw SRAM-PUF data from Arduino UNO with temperature and voltage variations and the code to read data from the .csv files. The zip file needs to be extracted to retrieve the sub-folders.

## ECC implementation 
### The ECC folder contains two subfolders i.e.: Soft-Decoding-UNO and Soft-Decoding-Zero. The code to implement soft decoding follows the algorithm proposed in the paper:
  Roel Maes et al. "Low-Overhead Implementation of a Soft Decision Helper Data Algorithm for SRAM PUFs" CHES, 2009.
  
### The soft decoding is built on top of the concatenated code - Repetition (3,1,3) + Reed-Muller (2,4) (Zero) or Reed-Muller(2,5) (UNO) proposed in the paper:
  Christoph Bosch et al. "Efficient Helper Data Key Extractor on FPGAs" CHES, 2008.
  
### Run the main file: 
"ECC/Soft-Decoding-UNO/Concat_code_softDecoding_ECC_withSRAMPUF.py" or "ECC/Soft-Decoding-Zero/Concat_code_softDecoding_ECC_withSRAMPUF.py" 

### The input binary files for the code are obtained from temperature sweep experiments on an Arduino UNO board with an SRAM PUF dimension of 128x64.
Input Dependencies: 
1) Reliability Information in the range of (50%,100%) for all PUF responses at ambient conditions - "Reliability_PUF_ambient_Br1.npy"
2) Corrected PUF responses using Transfer Learning at all temperatures - "TransferL_corrected_Resp_temp_all_Br1.npy"
3) Corrected PUF responses using Continual Learning at all temperatures - "ContinualL_corrected_Resp_temp_all_Br1.npy"
4) 
Golden Response at all temperatures in the range (-23.5C, 70C) - "GResp_temp_all_Br1.npy" required to compute BER, KER
### Soft Decoding Parameters for Concatenated Code(C1, C2):
C1 - Repetetion Code (n1,k1,d1); n1 = 3, k1 = 1, d1 = 3
C2 - Reed-Muller Code (n2,k2,d2); for RM(2,4) n2 = 16, k2 = 11, d2 = 4

Key Width = 128 bit, requiring ceil(171/(k1* k2))* n1* n2 = 768 raw PUF bits

## Min-Entropy Loss Computation
### The Min-Entropy-Loss folder contains the data files and code to implement the empirical min-entropy loss following the algorithm proposed in the paper:
  Jeroen Delvaux et al. "Efficient fuzzy extraction of PUF-induced secrets: Theory and applications" CHES, 2016.
  Min-entropy loss computed for BCH[15,11,1]; Standard array generated using all the codewords of BCH[15,11,1]

### Python Files to Compute the Min-entropy Loss: 
Dependencies shared for all: "Codeword_w.json" - all codewords for BCH[15,11,1] code & "std_array_p_complete.json" - complete standard array generated from the codewords
1) Min-entropy loss with fixed helper data: "entropy_BCH_15_11_1_fixed_helper_data_nolearning.py"
   Dependencies: Uniformity at all temperatures before using our proposed ML-assisted ECC - "Uniformity_pre_TL.npy"
2) Min-entropy loss with fixed helper data and Transfer Learning: "entropy_BCH_15_11_1_fixed_helper_data_w_TL.py"
   Dependencies: Uniformity at all temperatures after using our proposed ML-assisted ECC (Transfer Learning) - "Uniformity_post_TL.npy"
3) Min-entropy loss with multiple helper data: "entropy_BCH_15_11_1_multiple_helper_data.py"
   Dependencies: "Uniformity_pre_TL.npy" and Erroneous PUF responses before Transfer Learning at all temperatures for Arduino UNO board with an SRAM PUF dimension of 128x64 - "GResp_'+(Temp)+'.json"

### Python Files to plot Min-entropy Loss: 
To directly see the results use this file - Plot uniformity, min-entropy, and loss w.r.t. temperature: "Plot_all.py" using the output (.npy) files generated from the previous steps. 


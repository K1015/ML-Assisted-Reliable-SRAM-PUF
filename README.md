# ML-Assisted-Reliable-SRAM-PUF
Code for paper titled "Where are My Errors?: Unleashing Machine Learning to Stabilize SRAM-PUFs with Reduced Entropy Leakage" with the following functionalities:

## Soft decoding ECC implementation 

### The Soft-Decoding-Based-ECC folder contains the data files and code to implement soft decoding following the algorithm proposed in the paper:

  Roel Maes et al. "Low-Overhead Implementation of a Soft Decision Helper Data Algorithm for SRAM PUFs" CHES, 2009.
  
### The soft decoding is built on top of the concatenated code - Repetition (3,1,3) + Reed-Muller (2,4) proposed in the paper:

  Christoph Bosch et al. "Efficient Helper Data Key Extractor on FPGAs" CHES, 2008.

### Run the main file: 
"Soft-Decoding-based-ECC/Concat_code_softDecoding_ECC_withSRAMPUF.py" 

### The input binary files for the code are obtained from temperature sweep experiments on an Arduino UNO board with an SRAM PUF dimension of 128x64.
Dependencies: 
1) Majority Voted (over 15 measurements) Golden Response at all temperatures in the range (-23.5C, 70C) - "GResp_temp_all_Br1.npy"
2) Reliability Information in the range of (50%,100%) for all PUF responses at ambient conditions - "Reliability_PUF_ambient_Br1.npy"
3) Corrected PUF responses using Transfer Learning at all temperatures - "TransferL_corrected_Resp_temp_all_Br1.npy"
4) Corrected PUF responses using Continual Learning at all temperatures - "ContinualL_corrected_Resp_temp_all_Br1.npy"

### Soft Decoding Parameters for Concatenated Code(C1, C2):
C1 - Repetetion Code (n1,k1,d1); n1 = 3, k1 = 1, d1 = 3

C1 - Reed-Muller Code (n2,k2,d2); for RM(2,4) n2 = 16, k2 = 11, d2 = 4

Key Width = 128 bit, requiring ceil(171/(k1* k2))* n1* n2 = 768 raw PUF bits

### Two output plots are:
'BER_SRAM_PUF_Temp_all.pdf' - BER vs Temp for all cases

'BER_SRAM_PUF_Temp_all.pdf' - KER vs Temp for all cases

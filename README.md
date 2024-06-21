# ML-Assisted-Reliable-SRAM-PUF
This repository contains the code for the paper titled "Enhancing SRAM-Based PUF Reliability Through Machine Learning-Aided Calibration Techniques". The codes and data required to replicate the plots in the paper will be available upon request for the camera-ready version. Please note that the raw data for the Zero boards are not included in this repository due to space limitations (CSV file size exceeds 100MB) but can be provided upon request.

This repository includes the following:
## Raw Data for Arduino UNO
The raw SRAM-PUF data from Arduino UNO with temperature and voltage variations and the code (Response_processing.py) to read data from the .csv files. 
The zip file needs to be extracted to retrieve the sub-folders: 
1) RoomTemp_UNO_10Boards_Temp
2) RoomTemp_UNO_8Boards_Volt
3) CollectedDataAcrossVoltage_UNO_8Boards
4) CollectedDataAcrossTemperature_UNO_10Boards

## ML-based PUF Recalibration
This has 4 sub-folders for Temperature-UNO, Temperature-Zero, Voltage-UNO, and Voltage-Zero. Each UNO directory contains data files obtained from the extraction in Raw_Data_UNO.zip. The ML-based recalibration code for Continual and Transfer Learning is inside each sub-directory. 

### Run the main file: 
1) Continual Learning: "MLbased_Recalibration_XX_Continual_Learning_YY.py"
2) Transfer Learning "MLbased_Recalibration_XX_Transfer_Learning_YY.py"

where XX is either UNO or Zero and YY is either Temp or Volt

### The input binary files for the code are obtained after ML-based PUF Recalibration. 
The SRAM PUF dimension in Arduino UNO boards is 128x64 and in Zero boards 6815x32.
Input Dependencies (all obtained from raw PUF processing): 
1) Reliability Information in the range of (50%,100%) for all PUF responses at ambient conditions: "Rel_UNO_AmbientTemp_nMeas15.npy" and "Rel_UNO_AmbientVolt_nMeas15.npy"; Zero: "Rel_temp_Zero_ambient_Oct14.npy" and "Rel_volt_Zero_ambient_Oct11.npy"
2) PUF Golden responses at all temperatures and all boards: "GResp_UNO_temp_all.npy" and "GResp_UNO_Volt_all.npy" ; Zero: "GResp_temp_Zero_all_Oct14.npy" and "GResp_volt_Zero_all_Oct11.npy"

Outputs:
1) Corrected PUF responses using Transfer Learning at all temperatures and all boards - UNO: "CorrectedResp_XX_TransferLearning_YY.npy" 
2) Corrected PUF responses using Continual Learning at all temperatures and all boards - UNO: "CorrectedResp_XX_ContinualLearning_YY.npy" 

## ECC implementation 
The ECC folder contains two subfolders i.e.: Soft-Decoding-UNO and Soft-Decoding-Zero, each of ECC on UNO and Zero boards respectively.

### The code to implement soft decoding follows the algorithm proposed in the paper:
  Roel Maes et al. "Low-Overhead Implementation of a Soft Decision Helper Data Algorithm for SRAM PUFs" CHES, 2009.
  
### The soft decoding is built on top of the concatenated code - Repetition (3,1,3) + Reed-Muller (2,4) (Zero) or Reed-Muller(2,5) (UNO) proposed in the paper:
  Christoph Bosch et al. "Efficient Helper Data Key Extractor on FPGAs" CHES, 2008.
  
### Run the main file: 
"ECC/Soft-Decoding-UNO/Concat_code_softDecoding_ECC_withSRAMPUF.py" or "ECC/Soft-Decoding-Zero/Concat_code_softDecoding_ECC_withSRAMPUF.py" 

### The input binary files for the code are obtained after ML-based PUF Recalibration. 
The SRAM PUF dimension in Arduino UNO boards is 128x64 and in Zero boards 6815x32.
Input Dependencies: 
1) Reliability Information in the range of (50%,100%) for all PUF responses at ambient conditions: "Rel_UNO_AmbientTemp_nMeas15.npy" and Zero: "Rel_temp_Zero_ambient_Oct14.npy"
2) Corrected PUF responses using Transfer Learning at all temperatures and all boards: "CorrectedResp_UNO_TransferLearning_Temp.npy" and Zero: "CorrectedResp_Zero_Temp_TransferLearning.npy"
3) Corrected PUF responses using Continual Learning at all temperatures and all boards: "CorrectedResp_UNO_ContinualLearning_Temp.npy" and Zero: "CorrectedResp_Zero_Temp_ContinualLearning.npy"
4) PUF Golden responses at all temperatures and all boards: "GResp_UNO_temp_all.npy" and Zero: "GResp_temp_Zero_all_Oct14.npy"
   
### Soft Decoding Parameters for Concatenated Code(C1, C2):
C1 - Repetetion Code (n1,k1,d1); n1 = 3, k1 = 1, d1 = 3
C2 - Reed-Muller Code (n2,k2,d2); for RM(2,4) n2 = 16, k2 = 11, d2 = 4 for Zero
C2 - Reed-Muller Code (n2,k2,d2); for RM(2,5) n2 = 32, k2 = 16, d2 = 8 for UNO

Key Width = 128 bit, requiring ceil(171/(k1* k2))* n1* n2 = 768 raw PUF bits for UNO and 1056 for Zero

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


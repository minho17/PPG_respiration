# PPG_respiration
This code is provided to reproduce the operation of the proposed framework described in the paper, "Deep Learning-Based Ensemble Framework for Estimating Respiration Rate from PPG".  
The provided package includes pretrained models required to evaluate data from a subset of subjects in the BIDMC database, allowing users to run the proposed framework and verify its performance.

1. Testing method
 - make_data1.m : preprocessing the data into a format compatible with the proposed framework (the CapnoBase or BIDMC dataset must be acquired in advance to run this code)
 - test.py : code for running the proposed framework

2. Performance verification  
  'result.txt' file shows the simulation results as follows:
 
   Data: 1 / win_anal: 960 / win_move: 90  
   Sub_7: 1.012 / 0.887 / 1.000 / 1.050 / 3.837 / 0.625 / 0.550
  
   This result correspond to Subject 7 from Data1 (BIDMC), and the seven scores represent the performance of each model as shown below.  
   [base model1 / base model2 / base model3 / HRR model / LRR model / ESF / ESF + AMS]  
   For detailed descriptions of each model and method, please refer to the paper.  
  
   This simulation was conducted under the following conditions.
   - Matlab R2023a
   - Python 3.9.12
   - Pytorch 1.10.0
  
  
   

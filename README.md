# MMAI 894 DL Project: Youtube Face_Rec

Data from http://www.cs.tau.ac.il/~wolf/ytfaces/index.html#overview

## Getting Started

Preliminary model code & procedure:
1. Run Face_Crop_Cascade_Selection.py to:
    1. Pick best face cascade
    2. Store recognizable frame address in disk (Suceed_identify.txt)
2. Run main_data_prep.py to:
    1. Split frames of each person by folder
    2. Shrink size
    3. Store train & test address in disk (Train_addrs.txt & Test_addrs.txt)
3. Run main_data_store.py to:
    1. Initiate HDF5 shell
    2. Crop faces and fill train & test data into HDF5. (With winning face_cascade)
4. Run main_model.py to run a preliminary CNN model
5. Run Hyper-tune.py to perform hyper-parameter tuning
	

Model Rework & Procedure:
1. Run main_data_store_modi.py:
    1. Read from Suceed_identify.txt for all candidate frames
    2. Run shrinkage to reduce size
    3. Store data in HDF5
2. Run main_model_modi.py (minor changes from previous)
3. Run Hyper-tune_modi.py (minor changes from previous)

Supporting file:
	Data_Prep.py 


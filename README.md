# MMAI 894 DL Project: Youtube Face_Rec

Data from http://www.cs.tau.ac.il/~wolf/ytfaces/index.html#overview

## Getting Started

1. Run Face_Crop_Cascade_Selection,py will compare three face_rec_cascade of cv2. Save the winning result (recognizable frames address) to a txt file. Suceed_identify.txt
2. Run main_data_prep.py will return train and test data address (with shrinkage). Save the restult to txt files. Train_addrs.txt, Test_addrs.txt
3. Run main_data_store.py will create a HDF5 shell, run face_rec_cascade to crop faces, store the faces into the HDF5 shell. Store data in dataset.hdf5. With ["train_img"] ["test_img"] ["train_labels"] ["test_labels"]
4. Run main_model.py will create and run a preliminary CNN model.
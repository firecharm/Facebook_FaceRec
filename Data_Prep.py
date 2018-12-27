import csv
import re
import os
import random

# Create full name list
def identity_name(database_folder, error_list):
    name_list = set()
    # Use os.walk to extract necessary info
    for root, dirs, files in os.walk(database_folder):
    # Root will be the whole path, Files will be the file name with extension
        for file in files:
            if file.endswith(".jpg"):
                name = root.split('/')[8] # This will be the person's name
                if name not in error_list: # Error_list from data cleaning
                    name_list.add(name)
    return list(name_list)

# A .csv file was provided named “YTFErrors.csv”, the file listed 55 names with at least one folder(video) which does not belong to that name.
# We start with creating a list of these names.

def data_clean(error_csv):
    error_list = []
    with open(error_csv) as csvfile:
        csv_reader = csv.reader(csvfile)  
    # eg. 'Alvaro_Silva_Calderon/0','Gary_Paer', 'Hank_McKinnell 0, 3, 4'
        for row in csv_reader:  
            error_list.append(row[0]) 
    # eg. 'Alvaro_Silva_Calderon/0','Gary_Paer', 'Hank_McKinnell 0'

    # Crete a list of just names
    error_list_temp = []
    for error in error_list:
        error_list_temp.append(re.split(r'\W+', error)[0])
    # eg. 'Alvaro_Silva_Calderon','Gary_Paer', 'Hank_McKinnell'

    return error_list_temp

# Create train and test folder path
def train_test_split(name_list):
    # List first folders of each person as the training data
    temp_list = list(name_list)
    train_set = set()
    for root, dirs, files in os.walk("/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/frame_images_DB"):
        for file in files:
            if file.endswith(".jpg"):
                name = root.split('/')[8]            
                if name in temp_list:
                    train_set.add(root)
                    temp_list.remove(name)
    
    # list second folders of applicable person as the testing data
    temp_list = list(set(name_list))
    test_set = set()
    for root, dirs, files in os.walk("/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/frame_images_DB"):
        for file in files:
            if file.endswith(".jpg"):
                name = root.split('/')[8]            
                if (name in temp_list) & (root not in train_set):
                    test_set.add(root)
                    temp_list.remove(name)

    return (train_set,test_set)
    
#  randomly select frames
def shrinkage(folder_path,num_frames):
    # For Train_set Randomly select frames, 48 per person
    # For Test_set Randomly select frames, 10 per person
    data_addrs = []
    for path in folder_path:
        file = random.sample(os.listdir(path),num_frames)

        for i in file:
            # This address will be the full address
            data_addrs.append(path+'/'+i)

    return data_addrs
    
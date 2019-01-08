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

def name_to_address(database_folder,name_list):
    temp_list = list(set(name_list))
    folders = set()
    for root, dirs, files in os.walk(database_folder):
        for file in files:
            if file.endswith(".jpg"):
                name = root.split('/')[8]            
                if name in temp_list:
                    folders.add(root)
    addrs = []
    for path in folders:
        for i in os.listdir(path):
            addrs.append(path+'/'+i)
    return (addrs)

# Create train and test folder path
def train_test_split(addrs_list):
    # Train
    temp_name_set = set()
    train_folder_dict = {}
    train_addrs = []
    for i in addrs_list:
        name = i.split("/")[-3]
        folder = i.split("/")[-3]+ "/"+ i.split("/")[-2]
        
        if name in temp_name_set:
            pass
        else:
            temp_name_set.add(name)
            train_folder_dict[folder] = 0

        if folder in train_folder_dict.keys():
            train_folder_dict[folder] += 1 # keep record of number of frames
            train_addrs.append(i)

    # Drop the person if number of frames is less than 30
    unwant_name = [] # Create list of unwanted
    for key, value in train_folder_dict.items():
        if value <30:
            unwant_name.append(key.split("/")[0])
    # If address folder not in unwant, then keep the address
    train_addrs_update = []
    for i in train_addrs:
        name = i.split("/")[-3]
        folder = i.split("/")[-3]+ "/"+ i.split("/")[-2]
        if name not in unwant_name:
            train_addrs_update.append(i)
    
    # Test
    test_addrs = []
    for i in addrs_list:
        name = i.split("/")[-3]
        folder = i.split("/")[-3]+ "/"+ i.split("/")[-2]
        # Constraint: 1. name not in unwant_name 2. folder not in train_folder
        if (name not in unwant_name) & (folder not in train_folder_dict.keys()) & (i not in train_addrs_update):
            test_addrs.append(i)

    return (train_addrs_update,test_addrs)
    
#  Shrink data to desired num of frames
def shrinkage(data_addrs,num_frames):
    folder_dict = {}
    addrs_list = []
    for i in data_addrs:
        folder = folder = i.split("/")[-3]+ "/"+ i.split("/")[-2]
        # Set initial number to be 0
        if folder not in folder_dict.keys():
            folder_dict[folder] = 0
        # Updating the num of frames till meet desired
        if folder_dict[folder] < num_frames:
            addrs_list.append(i)
            folder_dict[folder] += 1

    return(addrs_list)
    
# pylint: disable=no-member
import cv2
from Data_Prep import data_clean,identity_name,name_to_address

# We will flag a suceed face crop only if 1. found eye in the face 2. only one face exist
def run_cascade(face_cascade,addrs):
    face_num=0 # num of suceed face detection
    Suceed_addrs = [] # 
    eye_cascade = cv2.CascadeClassifier('cascade/haarcascade_eye.xml')

    for file in addrs:
        image = cv2.imread(file)
        image = cv2.resize(image,(300,300))
        gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Set the default scaleFactor to be 1.1, minNeighbors to be 5
        face = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5
    #             ,minSize=(30, 30)
        )
        eye_num = 0
        
        # Find eye
        for (x,y,w,h) in face:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eye_num += len(eyes)
            
        if ((len(face) == 1)) & (eye_num > 0):
            face_num += 1 
            Suceed_addrs.append(file)

    

    return (face_num,Suceed_addrs)

if __name__ == "__main__":
    error_list = data_clean("YTFErrors.csv")
    name_list = identity_name("frame_images_DB", error_list)
    addrs = name_to_address("frame_images_DB", name_list)
    # this will return all address of all frames (cleaned)
    print("Number of frames in total: ",len(addrs))
    # 587,765 frames in total

    # Start to compare cascades:
    cascadePath = "cascade/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)
    har_default, addrs_default = run_cascade(face_cascade,addrs)
    print("Har_default Suceed identify:", har_default)
    # Har_default identified 337552 faces

    cascadePath = "cascade/haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)
    har_alt, addrs_alt = run_cascade(face_cascade,addrs)
    print("Har_alt Suceed identify:", har_alt)
    # Har_alt identified 365872 faces

    cascadePath = "cascade/haarcascade_frontalface_alt_tree.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)
    har_alt_tree, addrs_alt_tree = run_cascade(face_cascade,addrs)
    print("Har_alt_tree Suceed identify:", har_alt_tree)
    # Har_alt_tree identified 31749 faces

    # Winner Cascade is Har_alt, with 365872 faces (62.2%)

    face_nums = (har_default,har_alt,har_alt_tree)
    address_lists = (addrs_default,addrs_alt,addrs_alt_tree)
    # Find the best cascade and its corresponding address
    result = address_lists[face_nums.index(max(face_nums))]

    # For persons with less than 30 frames, drop them from the list:
    test_name_list = []
    for i in result:
        test_name_list.append(i.split("/")[-3])

    # Create a list of names to be kept:
    test_name_list_copy = []
    for i in set(test_name_list):
        if test_name_list.count(i) >= 50:
            test_name_list_copy.append(i)

    # Create a copy (cleaned) of result:
    result_copy = []
    for i in result:
        if i.split("/")[-3] in test_name_list_copy:
            result_copy.append(i)
    # Cleaned result has 358,219 frames, 1084 names

    # Save the cleaned result to disk
    with open('Suceed_identify.txt', 'w') as f:
        for item in result_copy:
            f.write("%s\n" % item)

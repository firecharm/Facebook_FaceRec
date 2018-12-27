# following comment is needed for VScode:
# pylint: disable=no-member
import cv2

def face_crop_to_array(img_addrs,shape0,shape1):
    
    # Create a CascadeClassifier Object
    cascadePath = "/Users/yaoyucui/Works/Smith/Deep Learning/Youtube Dataset/cascade/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascadePath)

    image = cv2.imread(img_addrs)
    # Resize shape to desired
    image = cv2.resize(image,(shape0,shape1))
    # Cascade require gray scale input
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30))

    print(face)
#     x = face[0][0] # X coordinate of bottom left
#     y = face[0][1] # Y coordinate of bottom left
#     w = face[0][2] # Width
#     h = face[0][3] # Height

#     face_img  = image[y:(y+h), x:(x+w)]  

#     return face_img


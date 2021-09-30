import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')





#load function
def face_extractor(img):
    faces = face_classifier.detectMultiScale(img,1.3,5)

    if faces is ():
        return None





    for (x,y,w,h) in faces:
        x=x-10
        y=y-10
        cropped_face = img[y:y+h+50, x:x+w+50]

    return cropped_face







#video camera ON
cap = cv2.VideoCapture(0)
count=0


while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count +=1
        face = cv2.resize(face_extractor(frame), (400,400))

        file_name_path = './Images/' + str(count) + '.jpg'       #saved the images in this folder
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1 , (0,255,0), 2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Human not found")
        pass
    if cv2.waitKey(1) == 13 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print ("Collecting samples complete")

import cv2
# if error pops out I have to disable Kapersky protection
# from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('RDJ.jpg')
webcam = cv2.VideoCapture(cv2.CAP_DSHOW)


while True:

    succesful_frame_read, frame = webcam.read()
    grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)

    cv2.imshow('Clever Programmer Face Detector', frame)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

webcam.release()

print("Code Completed")

'''
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(128, 256),randrange(128,256),randrange(128,256)),2)

# print(face_coordinates)

cv2.imshow('Clever Programmer Face Detector', img)
cv2.waitKey()
# cv2.destroyAllWindows()
'''




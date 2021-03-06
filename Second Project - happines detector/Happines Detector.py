# Creating Happines Detector
# Including neccesary libraries 
import cv2

# Loading Cascades

face_cascade = cv2.CascadeClassifier('C:\\Users\\shira\\OneDrive\\Desktop\\Learining Code\\Vision\\Second Project - happines detector\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\shira\\OneDrive\\Desktop\\Learining Code\\Vision\\Second Project - happines detector\\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('C:\\Users\\shira\\OneDrive\\Desktop\\Learining Code\\Vision\\Second Project - happines detector\\haarcascade_smile.xml')

# Creating detect function
def detect(gray ,original):
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for(x, y, w, h) in faces:
        cv2.rectangle(original, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = original[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 22)
        for(ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        smile = smile_cascade.detectMultiScale(roi_gray, 1.8, 24)
        for(sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)
    return original
    
# Using webcom for output
video_capture = cv2.VideoCapture(0)
while True:
    _, original = video_capture.read()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, original)
    cv2.imshow('Video', canvas)   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()
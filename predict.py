import numpy as np
from keras.models import model_from_json
import operator
import cv2
import time
import sys, os
import pyautogui as pag

json_file = open("model-bw.json", "r")
model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(model_json)
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

cap = cv2.VideoCapture(0)

categories = {0: 'NONE', 1: 'UP', 2: 'TWO', 3: 'RIGHT', 4: 'LEFT', 5: 'PALM'}

while True:
    _, frame = cap.read()
   
    frame = cv2.flip(frame, 1)
        
    x1 = int(0.5*frame.shape[1])
    y1 = 10
    x2 = frame.shape[1]-10
    y2 = int(0.5*frame.shape[1])
    
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)
  
    roi = frame[y1:y2, x1:x2]
       
    roi = cv2.resize(roi, (75, 75)) 
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _,roi=cv2.threshold(roi, 165, 255, cv2.THRESH_BINARY)
    cv2.imshow("test", roi)
    
    result = loaded_model.predict(roi.reshape(1, 75, 75, 1))
    prediction = {'NONE': result[0][0], 
                  'UP': result[0][1], 
                  'TWO': result[0][2],
                  'RIGHT': result[0][3],
                  'LEFT': result[0][4],
                  'PALM': result[0][5]}
  
    prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
    
    cv2.putText(frame, prediction[0][0], (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)    
    cv2.imshow("Frame", frame)
    
    if prediction[0][0]=='UP':
        pag.press('up')
    elif prediction[0][0]=='TWO':
        pag.press('down')    
    elif prediction[0][0]=='RIGHT':
        pag.press('right')
    elif prediction[0][0]=='LEFT':
        pag.press('left')
    elif prediction[0][0]=='PALM':
        pag.press('space')
        time.sleep(1)
    
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break
        
 
cap.release()
cv2.destroyAllWindows()

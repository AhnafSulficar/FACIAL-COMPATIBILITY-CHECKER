from flask import Flask, render_template, Response, request,redirect,url_for
import cv2
import datetime, time
import os, sys,glob,shutil
import numpy as np
from threading import Thread
import face_recognition
import logging

lss = os.listdir("./static") 
table_data=[]
global capture,rec_frame, grey, switch, neg, face, rec, out,save_images,image_count,x
capture=0
grey=0
neg=0 
face=0
switch=1
rec=0
save_images=False
image_count=0

#make shots directory to save pics
try:
    os.mkdir('./static')
except OSError as error:
    pass


#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)

def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))   
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:            
            return frame           

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame=frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = ( int(w * r), 480)
        frame=cv2.resize(frame,dim)
    except Exception as e:
        pass
    return frame

def compare_faces(image1_path, image2_path):
    try:
        image1 = face_recognition.load_image_file(image1_path)
        image2 = face_recognition.load_image_file(image2_path)

        features = face_recognition.face_encodings(image1)[0]

        new_image_features = face_recognition.face_encodings(image2)[0]

        euclidean_distance = face_recognition.face_distance([features], new_image_features)
        similarity_score = 100 - (euclidean_distance * 100)
        return similarity_score
    except IndexError as e:
        logging.error(f"Could not recognize a face in one or both of the images: {str(e)}")
        return None

    except Exception as e:
        logging.error(f"An error occurred during face comparison: {str(e)}")
        return None


def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame,save_images,image_count,x
    x=0
    lst = os.listdir("./static") 
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)
            if save_images and image_count<2:
                image_path="1.JPG" if image_count==0 else "2.JPG"  
                cv2.imwrite(image_path,frame)
                image_count+=1
                
            if image_count==2:
                x=2
                save_images=False
                image_count=0          
            
                
            try:
                ret, buffer = cv2.imencode('.JPG', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/files',methods=['POST','GET'])
# def files():
#         folder = r'static'
#         table_data=[]
#         for filename in os.listdir(folder):
#                 table_data.append(filename)
#         return render_template('index2.html', table_data=table_data)
    
@app.route('/form',methods=['POST','GET'])
def form():
    global coo
    coo=0
    if request.method =="POST":
        return render_template('index2.html')
    elif request.method=="GET":
        coo+=1
        if coo==2:
            return redirect(url_for('form'))
        else:
            return render_template('index3.html')
    

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,save_images,image_count,x
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture,image_count
            capture=1
            save_images=True
        if x==2:
            similarity_score=compare_faces("./1.JPG","./2.JPG")
            print("Simalarity score:",similarity_score)
            x=0
            return render_template('index2.html',similarity_score=similarity_score[0])
          
            

        elif  request.form.get('next') == 'Next':
            return redirect(url_for('form'))
        
        elif  request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                #Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
    
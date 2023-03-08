from pythonosc import dispatcher
from pythonosc import osc_server
import cv2
import tensorflow as tf
import numpy as np


cats = ['closeup','medium','wide']
ip = '127.0.0.1'
recievePort = 8000

#load model
model = tf.keras.models.load_model('firstconvnet.model')

def prep(filepath):
    img_size=100
    img=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f'no image, path:{filepath}')
    else:
        resized = cv2.resize(img,(img_size,img_size))
        return resized.reshape(-1,img_size,img_size,1)

def fromMax(address,*args):
    print('msg recieved from max')
    # pred = model.predict(prep('testingSS/3.png'))
    # print(cats[np.argmax(pred)])
    print(args[0])


def fromUnreal(address,msg):
    #print('msg recieved from unreal')
    fp = "oscTests/ss"+ str(msg) +".png"
    print(fp)
    pred = model.predict(prep(fp))
    print(cats[np.argmax(pred)])


#catches messages
dispatcher = dispatcher.Dispatcher()
dispatcher.map('/fromUnreal',fromUnreal)
dispatcher.map('/fromMax',fromMax)

#server
server = osc_server.BlockingOSCUDPServer((ip,recievePort),dispatcher)
print(f"server listening on {server.server_address}")
server.serve_forever()
#%%
import numpy as np
import pyautogui 
import matplotlib.pyplot as plt
from skimage import color,exposure
import skimage
from PIL import ImageGrab,Image
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from collections import deque
import random
import time
import tensorflow as tf
import win32gui
import win32api
import win32con
from warnings import filterwarnings
filterwarnings("ignore" )

#%%

#while True:
#    print(pyautogui.position())

#%%
import keras


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4 } ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

try:
    window = win32gui.FindWindow(None,r"T-Rex Run! - Chrome Dinosaur Game - Google Chrome")
except:
    pass

windowChild = win32gui.GetWindow(window,win32con.GW_CHILD)


NBACTIONS = 2
IMGHEIGHT = 120
IMGWIDTH  = 240
IMGHISTORY = 4

OBSERVEPERIOD = 64
GAMMA = 0.85
BATCH_SIZE = 32

ExpReplay_CAPACITY = 3000

game_coords = [100,380,860,562]
score_coords= [760,355,860,385]
try_again   = [480,470]

score = 0

#%%
def ProcessGameImage(RawImage):
    RawImage = np.array(RawImage)
    if RawImage.shape[2] == 3:
        GreyImage = color.rgb2gray(RawImage)
    
    ReducedImage = skimage.transform.resize(GreyImage,(IMGHEIGHT,IMGWIDTH))
    ReducedImage =  exposure.rescale_intensity(ReducedImage,out_range=(0,255))
    
    ReducedImage = ReducedImage / 255
    
    return ReducedImage

def findScore(RawImage):
    RawImage = np.array(RawImage)
    score = cv2.cvtColor(RawImage, cv2.COLOR_BGR2GRAY)
    score = cv2.bitwise_not(score)
    #plt.imshow(score)
    #plt.show()
    score = pytesseract.image_to_string(score,config="digits")
    if score == "":
        score = 0
    return score
    
#%%

class Agent:
    
    def __init__(self):
        
        self.model = self.createModel()        
        self.ExpReplay = deque()     
        self.steps = 0      
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
    
    def createModel(self):
        
        model = Sequential()
        model.add(Conv2D(filters=8,kernel_size=(2,2),
                         strides=(1,1),
                         input_shape=(IMGHEIGHT,IMGWIDTH,IMGHISTORY),
                         padding="same",activation="relu"))
        model.add(Conv2D(filters=8,kernel_size=(2,2),
                         strides=(1,1), padding="same",
                         activation='relu'))
        model.add(MaxPooling2D(2,2))
        
        model.add(Flatten())
        
        model.add(Dense(units=64,activation="relu"))
        model.add(Dense(units=NBACTIONS,activation="linear"))
        
        model.compile(loss="mse",optimizer="adam")
        
        return model
    
    def FindBestAct(self,s):
        if random.random() < self.epsilon or self.steps < OBSERVEPERIOD:
            return random.randint(0,NBACTIONS - 1)
        else:
            qvalue = self.model.predict(s)
            bestA = np.argmax(qvalue)
            return bestA
    
    def CaptureSample(self,sample):
        self.ExpReplay.append(sample)
        if len(self.ExpReplay) > ExpReplay_CAPACITY:
            self.ExpReplay.popleft()
        
        self.steps += 1
        
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon*self.epsilon_decay
    
    def Process(self):
        if self.steps > OBSERVEPERIOD:
            minibatch = random.sample(self.ExpReplay,BATCH_SIZE)
            batchlen = len(minibatch)
            
            inputs = np.zeros((BATCH_SIZE,IMGHEIGHT,IMGWIDTH,IMGHISTORY))
            targets = np.zeros((inputs.shape[0],NBACTIONS))
            
            Q_sa = 0
            
            for i in range(batchlen):
                state_t =  minibatch[i][0]
                action_t = minibatch[i][1]
                reward_t = minibatch[i][2]
                state_t1 = minibatch[i][3]
                
                inputs[i:i + 1] = state_t
                targets[i] = self.model.predict(state_t)
                Q_sa = self.model.predict(state_t1)
                
                if state_t1 is None:
                    targets[i,action_t] = reward_t  
                else:
                    targets[i,action_t] = reward_t + GAMMA*np.max(Q_sa)
            self.model.fit(inputs,targets,batch_size=BATCH_SIZE,epochs=1,verbose=0)

def move(BestAction):
    global score
    if BestAction == 0:
        pass
    if BestAction == 1:
        #pyautogui.press("space")   
        temp = win32api.PostMessage(windowChild,win32con.WM_KEYDOWN,0x20,0)
        time.sleep(.05)
        temp = win32api.PostMessage(windowChild,win32con.WM_CHAR,0x20,0)
def getAtt():
    global score
    image = ImageGrab.grab(game_coords)
    image = ProcessGameImage(image)
    score1 = ImageGrab.grab(score_coords)
    score1 = str(findScore(score1))
    #print(score1)
    score1 = score1.split("-")
    #print(score1)
    score1 = "".join(score1)
    try:
        score1 = abs(int(score1))
        score = score1
    except:
        #print(score)
        return score,image.reshape(IMGHEIGHT,IMGWIDTH,1)
    #print(score)
    return score,image.reshape(IMGHEIGHT,IMGWIDTH,1)
        
#%%
train_time = 200000

def TrainExperiment():
    global score
        
    TrainHistory = []
      
    TheAgent = Agent()

    BestAction = 1
    
    [InitialScore,NewGameImage] = getAtt()
    move(BestAction)
    
    GameState = np.stack((NewGameImage,NewGameImage,NewGameImage,NewGameImage),axis=2)
    
    GameState = GameState.reshape(1,GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    for i in range(train_time):
        zaman = time.time()
        BestAction = TheAgent.FindBestAct(GameState)
        [InitialScore,NewGameImage] = getAtt()
        #plt.imshow(NewGameImage.reshape(IMGHEIGHT,IMGWIDTH))
        #plt.show()
        if NewGameImage.reshape(IMGHEIGHT,IMGWIDTH)[IMGHEIGHT//2,IMGWIDTH//2] < 0.005:
            pyautogui.click(x=try_again[0],y=try_again[1])            
            pyautogui.click(x=360,y=900)
            time.sleep(0.5)
            zaman = time.time()
            [InitialScore,NewGameImage] = getAtt()
            move(BestAction)
        else:
            move(BestAction)  
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage,GameState[:,:,:,:3],axis=3)
        
        TheAgent.CaptureSample((GameState,BestAction,InitialScore,NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 1 == 0:
            print("Train Time: ",i," Game Score: ",score,"Time: ",time.time()-zaman)
            TrainHistory.append(score)
 #%%
time.sleep(2)
TrainExperiment()

#%%











#%%
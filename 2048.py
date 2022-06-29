#%%
import numpy as np
import pyautogui 
import matplotlib.pyplot as plt
from skimage import color
import skimage
from PIL import ImageGrab,Image
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten
from keras.layers.convolutional import Conv2D
from collections import deque
import random
import time
import tensorflow as tf
import keras
import tensorflow.keras.backend as K
tf.config.list_physical_devices('GPU') 
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4 } ) 
sess = tf.compat.v1.Session(config=config) 
keras.backend.set_session(sess)
#%%
"""
while True:
    print(pyautogui.position())"""
#GAME URL = "https://play2048.co/"
#%%
def ProcessGameImage(RawImage):
    RawImage = np.array(RawImage)
    if RawImage.shape[2] == 3:
        GreyImage = skimage.color.rgb2gray(RawImage)
    
    ReducedImage = skimage.transform.resize(GreyImage,(30,30))
    ReducedImage = skimage.exposure.rescale_intensity(ReducedImage,out_range=(0,255))
    
    ReducedImage = ReducedImage / 255
    
    return ReducedImage

def findScore(RawImage):
    RawImage = np.array(RawImage)
    score = cv2.cvtColor(RawImage, cv2.COLOR_BGR2GRAY)
    
    score = cv2.bitwise_not(score)
    score = pytesseract.image_to_string(score,config="digits")
    if score == "":
        score = 0
    return int(score)
    
#%%
NBACTIONS = 4
IMGHEIGHT = 30
IMGWIDTH  = 30
IMGHISTORY = 4

OBSERVEPERIOD = 100
GAMMA = 0.975
BATCH_SIZE = 64

ExpReplay_CAPACITY = 2000

game_coords = [228,308,714,791]
score_coords= [500,118,600,170]
try_again   = [470,601]
# GAME URL = "https://play2048.co/"

class Agent:
    
    def __init__(self):
        
        self.model = self.createModel()        
        self.ExpReplay = deque()     
        self.steps = 0      
        self.epsilon = 1.0
    
    def createModel(self):
        
        model = Sequential()
        
        model.add(Conv2D(filters=32,kernel_size=4,strides=(2,2),
                         input_shape=(IMGHEIGHT,IMGWIDTH,IMGHISTORY),
                         padding="same"))
        model.add(Activation("relu"))
        model.add(Conv2D(filters=64,kernel_size=4,strides=(2,2),
                         padding="same"))
        model.add(Activation("relu"))     
        
        model.add(Flatten())
        
        model.add(Dense(units=512))
        model.add(Activation("relu"))
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
        
        self.epsilon = 1.0
        if self.steps > OBSERVEPERIOD:
            self.epsilon = 0.75
        if self.steps > 200:
            self.epsilon = 0.5
        if self.steps > 400:
            self.epsilon = 0.25
        if self.steps > 1000:
            self.epsilon = 0.15
        if self.steps > 2500:
            self.epsilon = 0.1
        if self.steps > 3000:
            self.epsilon = 0.05
    
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
    if BestAction == 0:
        pyautogui.press("right")
    if BestAction == 1:
        pyautogui.press("left")
    if BestAction == 2:
        pyautogui.press("up")
    if BestAction == 3:
        pyautogui.press("down")
    time.sleep(0.25)
    image = ImageGrab.grab(game_coords)
    image = ProcessGameImage(image)

    score = ImageGrab.grab(score_coords)
    plt.imshow(score)
    score = findScore(score)
    return int(score),image.reshape(30,30,1)
        
#%%
train_time = 500

def TrainExperiment():
    
    TrainHistory = []
    
    
    TheAgent = Agent()

    BestAction = 0
    
    [InitialScore,InitialGameImage] = move(BestAction)
    
    GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis=2)
    
    GameState = GameState.reshape(1,GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    for i in range(20000):
        zaman = time.time()
        
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore,NewGameImage] = move(BestAction)
        print()
        if NewGameImage.reshape(30,30)[0,0] ==  0.8674115889559026:
            pyautogui.click(x=try_again[0],y=try_again[1])
            time.sleep(0.15)
            [ReturnScore,NewGameImage] = move(BestAction)
            
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage,GameState[:,:,:,:3],axis=3)
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 1 == 0:
            print("Train Time: ",i," Game Score: ",ReturnScore,"Time: ",time.time()-zaman)
            TrainHistory.append(ReturnScore)
#%%
time.sleep(2)
TrainExperiment()
#GAME URL = "https://play2048.co/"
#%%











#%%
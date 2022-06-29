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
from keras.layers.convolutional import Conv2D
from collections import deque
import random
import time
import tensorflow as tf

#%%

#while True:
#    print(pyautogui.position())

NBACTIONS = 2
IMGHEIGHT = 40
IMGWIDTH  = 40
IMGHISTORY = 4

OBSERVEPERIOD = 100
GAMMA = 0.975
BATCH_SIZE = 64

ExpReplay_CAPACITY = 4000

game_coords = [105,170,582,810]
score_coords= [300,240,394,312]
try_again   = [276,564]

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
    
#%%

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
        model.add(Conv2D(filters=32,kernel_size=2,strides=(2,2),
                         padding="same"))
        model.add(Activation("relu"))     
        
        model.add(Flatten())
        
        model.add(Dense(units=256))
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
        if self.steps > 600:
            self.epsilon = 0.5
        if self.steps > 1200:
            self.epsilon = 0.25
        if self.steps > 2000:
            self.epsilon = 0.15
        if self.steps > 4000:
            self.epsilon = 0.1
        if self.steps > 6000:
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
    global score
    if BestAction == 0:
        pass
    if BestAction == 1:
        pyautogui.press("space")        
def getAtt():
    global score
    image = ImageGrab.grab(game_coords)
    image = ProcessGameImage(image)
    score += 1
    return int(score),image.reshape(IMGHEIGHT,IMGWIDTH,1)
        
#%%
train_time = 2000

def TrainExperiment():
    global score
    
    TrainHistory = []
      
    TheAgent = Agent()

    BestAction = 1
    
    [InitialScore,InitialGameImage] = getAtt()
    move(BestAction)
    
    GameState = np.stack((InitialGameImage,InitialGameImage,InitialGameImage,InitialGameImage),axis=2)
    
    GameState = GameState.reshape(1,GameState.shape[0],GameState.shape[1],GameState.shape[2])
    
    for i in range(2000):
        BestAction = TheAgent.FindBestAct(GameState)
        [ReturnScore,NewGameImage] = getAtt()
        if NewGameImage.reshape(IMGHEIGHT,IMGWIDTH)[34,16] < 0.005 or NewGameImage.reshape(IMGHEIGHT,IMGWIDTH)[25,16] < 0.15 or NewGameImage.reshape(IMGHEIGHT,IMGWIDTH)[18,20] < 0.011 :
            pyautogui.click(x=try_again[0],y=try_again[1])            
            [InitialScore,InitialGameImage] = getAtt()
            move(BestAction)
            score = 0
            ReturnScore = 0
        else:
            move(BestAction)  
        NewGameImage = NewGameImage.reshape(1,NewGameImage.shape[0],NewGameImage.shape[1],1)
        
        NextState = np.append(NewGameImage,GameState[:,:,:,:3],axis=3)
        
        TheAgent.CaptureSample((GameState,BestAction,ReturnScore,NextState))
        
        TheAgent.Process()
        
        GameState = NextState
        
        if i % 1 == 0:
            print("Train Time: ",i," Game Score: ",ReturnScore)
            TrainHistory.append(ReturnScore)
 #%%
time.sleep(2)
TrainExperiment()

#%%











#%%
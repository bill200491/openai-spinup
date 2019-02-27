import os, sys, json
import tensorflow as tf
import pandas as pds
import numpy as np
import gym, cv2
from gym import spaces
from gym.utils import colorize, seeding, EzPickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from scipy.spatial import distance
from random import shuffle
import imutils

tar_ini=10.0

class guesture(gym.Env, EzPickle):
    
    def __init__(self, isTrain, path, subsets):
        EzPickle.__init__(self)
        self.images= n= len(subsets)
        
        inf= float('inf')
        a= [-inf]* n
        b= [inf]* n
        
        self.observation_space = spaces.Box( np.array(a), np.array( b), dtype=np.float32)
        # Action space ( EQ weight, HYG weight, TLT weight, Cash weight )
        self.action_space = spaces.Box(np.array([-1]*n), np.array([1]*n), dtype=np.float32)
        
        
        self.isTrain= isTrain
        self.path= path
        self.subsets= subsets
        self.width=128
        self.height=64
        self.resets= 0
        self.no_list= list( range(0, 10000))
        
    def reset(self):
        self.frames= 0
        self.pos= np.array([0.]*self.images)
        self.dst= tar_ini
        
        i= self.resets
        self.imgNo = k= self.no_list[i]% self.images
        self.resets = (self.resets +1)%10000
        #self.imgNo = k=np.random.randint(0, self.images)
        sb= np.random.randint(0, self.subsets[k])
        
        fn= self.path + '/' + '%d-%d.mp4'%( k, sb)
        self.tar= np.array([0.]*self.images)
        self.tar[k]= tar_ini
        
        self.trans=[np.random.randint(0,100)-50, np.random.randint(0,100)-50 ]
        self.angle= np.random.randint(0,30)-15
        self.obs= np.concatenate( (self.pos, self.tar))
        #print( fn)
        self.cap = cv2.VideoCapture(fn)
        
        ret, frame= self.cap.read()
        moved= imutils.translate(frame, self.trans[0], self.trans[1])
        rotated = imutils.rotate( moved, self.angle)
        self.frame = cv2.resize(rotated, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        #print('---->', self.frames, k, self.pos, self.tar, self.dst)

        return self.frame
    
    #def step(self, a):
        #self.frames +=1
        ##a= np.clip(a, -1,1)
        #self.pos += a
        
        #dst = distance.euclidean( self.pos, self.tar)
        #r= self.dst - dst # distance should be shorter and shorter
        #k= self.imgNo
        #dk= self.pos[k] - self.tar[k]
        #i= np.argmax(self.pos)
        #if i==k:
            #r= dk
        #else:
            #r=  self.dst - dst
        #self.dst= dst
        
        #ret, frame= self.cap.read()
        #if ret==True:
            #self.frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        ##print( self.frames, r, self.pos, self.tar, 'pos:%d(%d)'%(i, k), dk)
        #if ret==False or 199<self.frames or 30<self.pos[i]:
            #cls=self.get_class()
            
            #if cls== self.imgNo:
                #r= 100.0
            #else:
                #r= -dst
                
            #print('%3d'%self.frames, r, self.pos, self.tar, 'pos:%d(%d)'%(i, k), dst, 'class:%d(%d)'%(cls, self.imgNo))
            ##print( '%3d pos(%7.2f):(%7.2f %7.2f) rwd:%7.2f class:%d(%d)'%(self.frames, d, self.x, self.y, r, cls, self.imgNo))
            #ret=False
    
    def step(self, a):
        self.frames +=1
        #a= np.clip(a, -1,1)
        self.pos += a
        delta= self.pos- self.tar
        #self.obs= np.concatenate( (self.pos, delta) )
        
        dst = distance.euclidean( self.pos, self.tar)
        r= self.dst - dst # distance should be shorter and shorter
        self.dst= dst
        
        ret, frame= self.cap.read()
        if ret==True:
            moved= imutils.translate(frame, self.trans[0], self.trans[1])
            rotated = imutils.rotate( moved, self.angle)
            self.frame = cv2.resize( rotated, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        #print( self.frames, r, self.pos, self.tar, self.dst, self.obs)
        #if ret==False or 199<self.frames or 30.0 <dst:
        if ret==False or 199<self.frames:
            cls=self.get_class()
            
            #if cls== self.imgNo:
                #r= 100.0
            #else:
                #r= -dst
                
            print('%3d'%self.frames, r, self.pos, self.tar, self.dst, 'class:%d(%d)'%(cls, self.imgNo))
            #print( '%3d pos(%7.2f):(%7.2f %7.2f) rwd:%7.2f class:%d(%d)'%(self.frames, d, self.x, self.y, r, cls, self.imgNo))
            ret=False
        
        return self.frame, r, not ret, self.pos
    
    def render(self):
        cv2.imshow('frame', self.frame)
        cv2.waitKey(20)
        
    def get_actions(self):
        a= self.tar/tar_ini
        return a
        
        
    def get_class(self):
        return np.argmax(self.pos)
        #n= self.images
        #bnd= (tar_ini/2.0)
        #for i in range(n):
            #tar=[0]*n
            #tar[i]= tar_ini
            #dst = distance.euclidean( self.pos, tar)
            ##print('--->',dst)
            #if dst< bnd:
                #return i
        #return -1
            
        

if __name__ == '__main__':
    
    env= guesture( True, '../trainset', [ 1, 1, 1, 1] )
    
    for i in range(10):
        f= env.reset()
        env.render()
        while True:
            #a= np.random.uniform(-1,1, 4)
            a= env.get_actions()
            f, r, done, pos= env.step(a)
            if done==True:
                break
            env.render()
            #cv2.imshow('act:%0.2f %0.2f R:%0.2f pos:%0.2f %0.2f'%( a[0], a[1], r,pos[0], pos[1]), f)
            #cv2.waitKey(10)
            #cv2.destroyAllWindows()
    #env.render()
        print('Class %d vs %d'%(env.get_class(), env.imgNo  ) )

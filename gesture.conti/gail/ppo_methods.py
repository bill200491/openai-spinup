import numpy as np
import tensorflow as tf
import gym
import time
#from spinup.guesture.environment import guesture
from guestenv import guesture
import core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs


# pkill -9 python3 ; python3 ppo.py --hid 100 --l 1  --steps 800 --epochs 10 --exp_name guesture_ppo --cpu 1

def show_result( env, sess, img_ph, x_ph, a_ph, mu, epochs):
    

    for i in range( 100*epochs):
        o= env.reset()
        p=np.array([0]* env.images)
        env.render()
        
        while True:
            p= np.array([0]* env.images)
            a= sess.run( mu, feed_dict={ img_ph:[o], x_ph: p.reshape(1,-1)})[0]
            #print(a)
            o, r, done, pos= env.step(a)
            if done==True:
                break
            env.render()
            
        #v=val
        #for j,r in enumerate(ra):
            #ra[j]= [v]
            #v -= r
        
        #for j,r in enumerate(ra):
            #print( j, r)
        #print('---->',val,v)
            
        #pos_list.append(pa)
        #act_list.append(aa)
        #val_list.append(ra)
    
    #pos_list= np.array(pos_list)
    #act_list= np.array(act_list)
    #val_list= np.array(val_list)
    
    #print(cls_list)
    #print(pos_list.shape, act_list.shape, val_list.shape )
    #np.savez( 'trajectories.npz', obs=np.array(pos_list), acs=np.array(act_list), ep_lens=np.array(cls_list), ep_rets=np.array(val_list))


    

def behavior_cloning( env, sess, img_ph, x_ph, a_ph, mu, v,  bh_loss, train_bh, epochs):
    #generate_behavior(env, epochs)
    
    
    img_list, obs_list, act_list= [], [], []
    for i in range( epochs):
        f= env.reset()
        pos= np.array([0]* env.images)
        
        while True:
            pos= np.array([0]* env.images)
            if np.random.randint(0,1)==0:
                a= env.get_actions()
            else:
                a= np.random.uniform(-1,1, 4)

            img_list.append( f)
            obs_list.append( pos)
            act_list.append( a)
            
            f, r, done, pos= env.step(a)
            
            if done==True:
                break
        
        if (i%64)==63:
            min_loss= float('inf')
            cnt=0
            
            batch_size=50
            batchs= len(img_list)// batch_size
            
            for j in range(100):
                loss=0
                for k in range(batchs):
                    p= k* batch_size
                    q= (k+1)* batch_size
                    
                    bimg= img_list[ p:q]
                    bobs= obs_list[ p:q]
                    bact= act_list[ p:q]
                    l= sess.run([train_bh, bh_loss], {img_ph:bimg, x_ph: bobs, a_ph:bact})[-1:]
                    
                    loss+= l[0]
                loss /= batchs 
                
                if loss < min_loss:
                    min_loss = loss
                    cnt =0
                elif 30<j:
                    cnt += 1
                    
                print('i;%d j:%d %d loss:%f(%f)'%( i, j, cnt,loss, min_loss))
                if 2<cnt:
                    break
                    
            img_list, obs_list, act_list= [], [], []
        
    #show_result( env, sess, img_ph, x_ph, a_ph, mu, epochs)    
    
    

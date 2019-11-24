"""
Created on Fri Apr 16 17:02:15 2019
track-feat-v1
When the reward from the env is less than or equal to 0, we think tracking failed.
So set the episode_over=True.

Consider the length limitation of the sequences, 20-40?
   episode_length, when tracking failed, set episode_over=True.
@author: qxy
"""

import gym
import numpy as np
import pickle
import random

import sys
sys.path.append('/home/dylan/DL/Codes/myTrack')
from scipy.misc import imread 
from baselines.common import tf_util as U
#from gailtf.baselines.common import set_global_seeds, tf_util as U

from gtutils import cal_distance, move_bbox, compute_iou, crop_image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

class TrackEnv(gym.Env):

    def __init__(self, db='VOT'):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=100, shape=(1024,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=-128, high=128, shape=(107,107,3))
        self.observation_space = gym.spaces.Tuple((
                gym.spaces.Box(low=-128, high=128, shape=(107,107,3), dtype=np.float32), 
                gym.spaces.Box(low=-128, high=128, shape=(107,107,3), dtype=np.float32)
                ))
        
        pkl_path = 'dataset/vot-otb.pkl' if db=='VOT' else 'dataset/otb-vot.pkl'
        self.data_path = "dataset/"
        with open(pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
        self.n_seq = len(self.dataset)
        self.seq_names = list(self.dataset)
  
        self.min_len = 40

    def step(self, action):
        
        self.pointer += 1
        idx = self.pointer
        
#        # limitation on action
#        ac[2] = np.clip(ac[2],-0.05,0.05)
#        ac[3] = np.clip(ac[3],-0.05,0.05)
        
        # compute reward of previous frame
        self.pos_trackerCurr = move_bbox(self.pos_trackerCurr, action, self.img.shape)
        reward = compute_iou(self.pos_trackerCurr, self.gts[idx-1])
                
        if idx==self.n_images or reward<=0.3:
            ob = None
            episode_over = True
            gt = None
        else:
            img_path = self.data_path + self.seq_id + r'/' + self.images[idx]
            self.img = imread(img_path, mode='RGB')
            ob = crop_image(self.img, self.pos_trackerCurr)
            self.img_g, self.img_l = ob[0], ob[1]
            episode_over = False
            gt = self.gts[idx]
            
        tracker_info = {'gt': gt, 'tracker_post': self.pos_trackerCurr}

        return ob, reward, episode_over, tracker_info

    def reset(self, seq_name=None):
        """ Repeats NO-OP action until a new episode begins. """
        self.seq_id = seq_name if seq_name else random.choice(self.seq_names)     
        self.gts = self.dataset[self.seq_id]['gt']
        self.images = self.dataset[self.seq_id]['images']
        self.n_images = len(self.images)
        assert(self.n_images == len(self.gts))

        self.pointer = idx = 1 + np.random.randint(max(1, self.n_images-self.min_len))

        img_path = self.data_path + self.seq_id + r'/' + self.images[idx]
        self.img = imread(img_path, mode='RGB')
        ob = crop_image(self.img, self.gts[idx-1])
        self.img_g, self.img_l = ob[0], ob[1]
               
        self.pos_trackerCurr = self.gts[idx-1]
        
        return ob

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        
        idx = self.pointer
        bbox_lastFrame = self.gts[idx]
        
        img_path = self.data_path + self.seq_id + r'/' + self.images[idx+1]
        img = imread(img_path, mode='RGB')         
        img_g, img_l = crop_image(img, bbox_lastFrame)
        
        plt.figure('img_g')
        plt.imshow(self.img_g)
        plt.pause(0.02)
        
        plt.figure('img_l')
        plt.imshow(self.img_l)
        plt.pause(0.02)

        
        plt.figure('img')
        plt.imshow(img)
        x, y, w, h = [int(round(num)) for num in self.pos_trackerCurr]
        currentAxis=plt.gca()
        rect=patches.Rectangle((x, y),w,h,linewidth=1,edgecolor='r',facecolor='none')
        currentAxis.add_patch(rect)            
        plt.show()
        plt.close('img_g')
        plt.close('img_l')
        plt.close('img')
        
    
    def gen_ob(self, idx, bbox_lastFrame):
        
        img_path = self.data_path + self.seq_id + r'/' + self.images[idx+1]    
        img_g, img_l = crop_image(imread(img_path, mode='RGB'), bbox_lastFrame)
        ob = self.featureEx.feature((img_g, img_l))
            
        return ob


if __name__ == '__main__':
    
    from track_policy import TrackPolicy, TrackPolicyNew
    
    env = TrackEnv()  
    actor = TrackPolicyNew("actor", 
                            ob_space=env.observation_space, 
                            ac_space=env.action_space)
    U.initialize()
    
    ob = env.reset('vot2016/hand')
    
    img = imread(env.data_path + env.seq_id + r'/' + env.images[0], mode='RGB')
        
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(img)
    

    gt = env.gts[0]
    gt_rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                            linewidth=2, edgecolor="r", zorder=1, fill=False)
    ax.add_patch(gt_rect)
    rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                             linewidth=2, edgecolor="g", zorder=1, fill=False)
    ax.add_patch(rect)
    
    plt.pause(.01)
    plt.draw()
    
    
#    ac1, vpred1 = actor.act(stochastic=False, fc_input=ob)
    ac1 = np.array(cal_distance(env.gts[0], env.gts[1]))
    #env.render()
    
    reward_sum = 0
    while True:
        
        im.set_data(env.img)
       
        gt = env.gts[env.pointer]
        gt_rect.set_xy(gt[:2])
        gt_rect.set_width(gt[2])
        gt_rect.set_height(gt[3])

        ob, reward, done, tracker_info = env.step(ac1)
            
        result_bb = tracker_info['tracker_post']
        rect.set_xy(result_bb[:2])
        rect.set_width(result_bb[2])
        rect.set_height(result_bb[3])
        
        plt.pause(.01)
        plt.draw()
        
        print(reward)
        reward_sum += reward

    
        if done:
            break
#        env.render()
        ac1, vpred1 = actor.act(stochastic=False, ob=ob)
#        ac1 = np.array(cal_distance(tracker_info['tracker_post'], tracker_info['gt']))
#        ac1 = np.array([0.05,0.05,0.00,0.00])
    print(reward_sum/(env.n_images-1))
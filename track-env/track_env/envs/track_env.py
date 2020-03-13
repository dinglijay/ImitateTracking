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
sys.path.append('../../..')
    
from PIL import Image
from boundingbox import BoundingBox, cal_distance, crop_resize
from conf.configs import ADNetConf

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class TrackEnv(gym.Env):

    def __init__(self, db='VOT', path_head='', data_path='dataset/'):

        super(TrackEnv, self).__init__()
        
        # Becasuse multiprocessing start env processings by 'spawn', 
        # ADNetConf configuration in the parent processing will be invalid
        ADNetConf.get('conf/dylan.yaml')
        self.stop_iou, self.stop_cnt = ADNetConf.get()['dl_paras']['stop_iou_cnt']
        self.sample_zoom = ADNetConf.g()['dl_paras']['zoom_scale']
        self.out_limit = ADNetConf.g()['dl_paras']['actor_out_limt']
        self.len_seq = ADNetConf.g()['dl_paras']['len_seq']
        self.reward_stages = ADNetConf.g()['dl_paras']['reward_stages']

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        ob_shape = (107, 107, 3) if len(self.sample_zoom)==1 else (len(self.sample_zoom),) + (107, 107, 3)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=ob_shape, dtype=np.uint8)

        self.data_path = data_path
        pkl_path = path_head + 'dataset/vot-otb.pkl' if db=='VOT' else path_head +'dataset/otb-vot.pkl'
        
        with open(pkl_path, 'rb') as f:
            self.dataset = pickle.load(f)
            
        self.n_seq = len(self.dataset)
        self.seq_names = list(self.dataset)
  
        self.min_len = 40

    def step(self, action):     
        self.pointer += 1
        idx = self.pointer
        
        # move bbox and compute reward
        action *= self.out_limit 
        pos_moved = self.pos_trackerCurr.move(action)
        iou = pos_moved.fit_image(self.img.size).iou(self.gts[idx-1])
        reward = cal_reward(iou, self.reward_stages)
        self.pos_trackerCurr = pos_moved.fit_image(self.img.size, margin=10)
        
        self.cnt_sml_rew = 0 if reward>self.stop_iou else self.cnt_sml_rew+1
        if idx==self.n_images or idx==self.stop_idx or self.cnt_sml_rew>self.stop_cnt :
            ob = None
            episode_over = True
            gt = None
        else:
            img_path = self.data_path + self.seq_id + r'/' + self.images[idx]
            self.img = Image.open(img_path)
            ob = crop_resize(self.img, self.pos_trackerCurr, zoom=self.sample_zoom)
            episode_over = False
            gt = self.gts[idx]
            
        tracker_info = {'gt': gt, 'tracker_post': self.pos_trackerCurr}

        return ob, reward, episode_over, tracker_info

    def reset(self, seq_name=None, startFromFirst=False):

        """ Repeats NO-OP action until a new episode begins. """
        self.seq_id = seq_name if seq_name else random.choice(self.seq_names)     
        self.gts = [BoundingBox(*i) for i in self.dataset[self.seq_id]['gt']]
        self.images = self.dataset[self.seq_id]['images']
        self.n_images = len(self.images)
        assert(self.n_images == len(self.gts))

        self.pointer = idx = 1 if startFromFirst else \
                             1 + np.random.randint(max(1, self.n_images-self.min_len))
        self.stop_idx = idx + self.len_seq

        img_path = self.data_path + self.seq_id + r'/' + self.images[idx]
        self.img = Image.open(img_path)
        ob = crop_resize(self.img, self.gts[idx-1], zoom=self.sample_zoom)
               
        self.pos_trackerCurr = self.gts[idx-1]
        self.cnt_sml_rew = 0
        
        return ob

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        return


def cal_reward(iou, stages=None):
    if not stages:
        return iou
    a, b, ar, br, cr = stages
    return float(np.piecewise(iou,[iou<a, a<=iou<=b, b<iou], [ar,br,cr]))


def main():
    
    from track_policy import TrackPolicy
    from baselines.common import tf_util as U
    
    ADNetConf.get('../../../conf/dylan.yaml')
    env = TrackEnv(db='OTB', path_head='../../../', data_path="../../../dataset/")
    ob = env.reset(startFromFirst=True)
    # ob = env.reset('vot2016/hand', startFromFirst=True)

    actor = TrackPolicy("actor",
                        ob_space=env.observation_space,
                        ac_space=env.action_space,
                        load_path='../../../log/0309_track2CnnFc12_noAct/checkpoints/01100')
    U.initialize()
    
    
    img = Image.open(env.data_path + env.seq_id + r'/' + env.images[0])
        
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    im = ax.imshow(img)
    
    gt = env.gts[0].to_tuple()
    gt_rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                            linewidth=2, edgecolor="r", zorder=1, fill=False)
    ax.add_patch(gt_rect)
    rect = plt.Rectangle(tuple(gt[:2]), gt[2], gt[3],
                             linewidth=2, edgecolor="g", zorder=1, fill=False)
    ax.add_patch(rect)
    
    plt.pause(.01)
    plt.draw()
    
    ac1, vpred1 = actor.act(stochastic=False, ob=ob)
    ac1 = np.clip(ac1, -1, 1)
    # ac1 = 5.0*np.array(cal_distance(env.gts[0], env.gts[1]))
    #env.render()
    
    reward_sum, cnt = 0, 0
    while True:
        
        im.set_data(env.img)
       
        gt = env.gts[env.pointer].to_tuple()
        gt_rect.set_xy(gt[:2])
        gt_rect.set_width(gt[2]-1)
        gt_rect.set_height(gt[3]-1)

        ob, reward, done, tracker_info = env.step(ac1)
        cnt +=1
            
        result_bb = tracker_info['tracker_post'].to_tuple()
        rect.set_xy(result_bb[:2])  
        rect.set_width(result_bb[2]-1)
        rect.set_height(result_bb[3]-1)
        
        plt.pause(.01)
        plt.draw()
        
        reward_sum += reward

        if done:
            break
#        env.render()
        ac1, vpred1 = actor.act(stochastic=False, ob=ob)
        ac1 = np.clip(ac1, -1, 1)

        # ac1 = 5.0*np.array(cal_distance(tracker_info['tracker_post'], tracker_info['gt']))
#        ac1 = np.array([0.05,0.05,0.00,0.00])
    print(reward_sum/(cnt))


def memory_analysis():
    
    # from memory_profiler import profile
    env = TrackEnv(path_head='../../../', data_path="../../../dataset/")

    for _ in range(2000):
        ob = env.reset()
        ac1 = np.array(cal_distance(env.gts[0], env.gts[1]))
        reward_sum, cnt = 0, 0

        while True:
            ob, reward, done, tracker_info = env.step(ac1)
            print(reward)
            reward_sum += reward
            cnt += 1
            if done:
                break
            ac1 = np.array(cal_distance(tracker_info['tracker_post'], tracker_info['gt']))
        print(cnt,reward_sum/(cnt))


if __name__ == '__main__':
    # memory_analysis()
    main()
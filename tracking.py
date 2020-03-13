from __future__ import division

import os
import random
import sys
import logging

import cv2
# import fire
import numpy as np
import tensorflow as tf
import time

import commons
from boundingbox import BoundingBox, Coordinate
from boundingbox import crop_resize, cal_distance
from conf.configs import ADNetConf
from track_policy import ADNetwork
from pystopwatch import StopWatchManager

_log_level = logging.DEBUG
_logger = logging.getLogger('ADNetRunner')
_logger.setLevel(_log_level)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(_log_level)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
_logger.addHandler(ch)

class ADNetRunner:
    MAX_BATCHSIZE = 512

    def __init__(self):
        self.tensor_input = tf.placeholder(tf.float32, shape=(None, 2, 107, 107, 3), name='patch')
        self.tensor_action_history = tf.placeholder(tf.float32, shape=(None, 40), name='action_history')
        self.tensor_lb_action = tf.placeholder(tf.float32, shape=(None, 4), name='lb_action')
        self.tensor_lb_class = tf.placeholder(tf.int32, shape=(None, ), name='lb_class')
        self.tensor_is_training = tf.placeholder(tf.bool, name='is_training')
        self.learning_rate_placeholder = tf.placeholder(tf.float32, [], name='learning_rate')

        self.persistent_sess = tf.Session(config=tf.ConfigProto(
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1
        ))

        self.adnet = ADNetwork(self.learning_rate_placeholder)
        self.adnet.create_network(self.tensor_input, 
                                  self.tensor_lb_action, 
                                  self.tensor_lb_class, 
                                  self.tensor_action_history, 
                                  self.tensor_is_training)
        if 'ADNET_MODEL_PATH' in os.environ.keys():
            self.adnet.read_original_weights(self.persistent_sess, os.environ['ADNET_MODEL_PATH'])
        else:
            self.adnet.read_original_weights(self.persistent_sess)

        self.action_histories = np.array([.0] * 40, dtype=np.float32)
        self.action_histories_old = np.array([.0] * 40, dtype=np.float32)
        self.histories = []
        self.iteration = 0
        self.imgwh = None

        self.callback_redetection = self.redetection_by_sampling
        self.failed_cnt = 0
        self.latest_score = 0

        self.stopwatch = StopWatchManager()

    def by_dataset(self, vid_path='./dataset/OTB/Human2'):
        assert os.path.exists(vid_path)

        gt_boxes = BoundingBox.read_vid_gt(vid_path)

        curr_bbox = None
        self.stopwatch.start('total')
        _logger.info('---- start dataset l=%d' % (len(gt_boxes)))
        for idx, gt_box in enumerate(gt_boxes):
            img = commons.imread(os.path.join(vid_path, 'img', '%04d.jpg' % (idx + 1)))
            self.imgwh = Coordinate.get_imgwh(img)
            if idx == 0:
                # initialization : initial fine-tuning
                self.initial_finetune(img, gt_box)
                curr_bbox = gt_box

            # tracking
            predicted_box = self.tracking(img, curr_bbox, gt_box)
            print(predicted_box.iou(gt_box))
            self.show(img, gt_box=gt_box, predicted_box=predicted_box)
            # cv2.imwrite('/Users/ildoonet/Downloads/aaa/%d.jpg' % self.iteration, img)
            curr_bbox = predicted_box
        self.stopwatch.stop('total')

        _logger.info('----')
        _logger.info(self.stopwatch)
        _logger.info('%.3f FPS' % (len(gt_boxes) / self.stopwatch.get_elapsed('total')))

    def show(self, img, delay=1, predicted_box=None, gt_box=None):
        if isinstance(img, str):
            img = commons.imread(img)

        if gt_box is not None:
            gt_box.draw(img, BoundingBox.COLOR_GT)
        if predicted_box is not None:
            predicted_box.draw(img, BoundingBox.COLOR_PREDICT)

        cv2.imshow('result', img)
        cv2.waitKey(delay)

    def _get_features(self, samples):
        feats = []
        for batch in commons.chunker(samples, ADNetRunner.MAX_BATCHSIZE):
            feats_batch = self.persistent_sess.run(
                    self.adnet.layer_feat, feed_dict={self.adnet.input_tensor: batch})
            feats.extend(feats_batch)
        return feats

    def initial_finetune(self, img, detection_box):
        self.stopwatch.start('initial_finetune')
        t = time.time()

        # generate samples
        pos_num, neg_num = ADNetConf.g()['initial_finetune']['pos_num'], ADNetConf.g()['initial_finetune']['neg_num']
        pos_boxes, neg_boxes = detection_box.get_posneg_samples(self.imgwh, pos_num, neg_num, use_whole=True)
        pos_lb_action = BoundingBox.get_action_labels_gail(pos_boxes, detection_box)

        feats = self._get_features([crop_resize(img, box.to_tuple()) for i, box in enumerate(pos_boxes)])
        for box, feat in zip(pos_boxes, feats):
            box.feat = feat
        feats = self._get_features([crop_resize(img, box.to_tuple()) for i, box in enumerate(neg_boxes)])
        for box, feat in zip(neg_boxes, feats):
            box.feat = feat

        # train_fc_finetune_hem
        self._finetune_fc(
            img, pos_boxes, neg_boxes, pos_lb_action,
            ADNetConf.get()['initial_finetune']['learning_rate'],
            ADNetConf.get()['initial_finetune']['iter']
        )

        self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))
        _logger.info('ADNetRunner.initial_finetune t=%.3f' % t)
        self.stopwatch.stop('initial_finetune')

    def _finetune_fc(self, img, pos_boxes, neg_boxes, pos_lb_action, learning_rate, n_iter, iter_score=1):
        BATCHSIZE = ADNetConf.g()['minibatch_size']

        def get_img(idx, posneg):
            if isinstance(img, tuple):
                return img[posneg][idx]
            return img

        pos_samples = [crop_resize(get_img(i, 0), box.to_tuple()) for i, box in enumerate(pos_boxes)]
        neg_samples = [crop_resize(get_img(i, 1), box.to_tuple()) for i, box in enumerate(neg_boxes)]
        # pos_feats, neg_feats = self._get_features(pos_samples), self._get_features(neg_samples)

        commons.imshow_grid('pos', pos_samples[-50:], 10, 5)
        commons.imshow_grid('neg', neg_samples[-50:], 10, 5)
        cv2.waitKey(1)

        for i in range(n_iter):
            batch_idxs = commons.random_idxs(len(pos_boxes), BATCHSIZE)
            batch_feats = [x.feat for x in commons.choices_by_idx(pos_boxes, batch_idxs)]
            batch_lb_action = commons.choices_by_idx(pos_lb_action, batch_idxs)
            self.persistent_sess.run(
                self.adnet.weighted_grads_op1,
                feed_dict={
                    self.adnet.layer_feat: batch_feats,
                    self.adnet.label_tensor: batch_lb_action,
                    self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 40)),
                    self.learning_rate_placeholder: learning_rate,
                    self.tensor_is_training: True
                }
            )
                          
            # scores_batch_pos = self.persistent_sess.run(
            #     self.adnet.layer_scores,
            #     feed_dict={
            #         self.adnet.layer_feat: batch_feats,
            #         self.adnet.action_history_tensor: np.zeros(shape=(BATCHSIZE, 40)),
            #         self.learning_rate_placeholder: learning_rate,
            #         self.tensor_is_training: False
            #     }
            # )

            if i % iter_score == 0:
                # training score auxiliary(fc2)
                # -- hard score example mining
                scores = []
                for batch_neg in commons.chunker([x.feat for x in neg_boxes], ADNetRunner.MAX_BATCHSIZE):
                    scores_batch = self.persistent_sess.run(
                        self.adnet.layer_scores,
                        feed_dict={
                            self.adnet.layer_feat: batch_neg,
                            self.adnet.action_history_tensor: np.zeros(shape=(len(batch_neg), 40)),
                            self.learning_rate_placeholder: learning_rate,
                            self.tensor_is_training: False
                        }
                    )
                    scores.extend(scores_batch)
                desc_order_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x:x[1][1])]

#                # -- train
                batch_feats_neg = [x.feat for x in commons.choices_by_idx(neg_boxes, desc_order_idx[:BATCHSIZE])]
                self.persistent_sess.run(
                    self.adnet.weighted_grads_op2,
                    feed_dict={
                        self.adnet.layer_feat: batch_feats + batch_feats_neg,
                        self.adnet.class_tensor: [1]*len(batch_feats) + [0]*len(batch_feats_neg),
                        self.adnet.action_history_tensor: np.zeros(shape=(len(batch_feats)+len(batch_feats_neg), 40)),
                        self.learning_rate_placeholder: learning_rate,
                        self.tensor_is_training: True
                    }
                )
    
    def tracking(self, img, curr_bbox, gt_box):
        self.iteration += 1
        is_tracked = True
        self.latest_score = -1
        self.stopwatch.start('tracking.do_action')
        
        print(self.iteration)
    
        patch = crop_resize(img, curr_bbox.to_tuple())

        # forward with image & action history
        actions, classes = self.persistent_sess.run(
            [self.adnet.layer_actions, self.adnet.layer_scores],
            feed_dict={
                self.adnet.input_tensor: [patch],
                self.adnet.action_history_tensor: [self.action_histories],
                self.tensor_is_training: False})
        # actions = np.array(cal_distance(curr_bbox, gt_box))[None]
        actions[:,2:][abs(actions[:,2:])>0.15] = 0
        
        self.action_histories_old = np.copy(self.action_histories)
        self.action_histories = np.insert(self.action_histories, 0, actions[0])[:-4]

        self.latest_score = latest_score = classes[0][1]
        if latest_score < ADNetConf.g()['predict']['thresh_fail']:
            is_tracked = False
        else:
            self.failed_cnt = 0
        
        # move box
        curr_bbox = curr_bbox.do_action_gail(actions, self.imgwh)
        self.stopwatch.stop('tracking.do_action')

        # redetection when tracking failed
        new_score = 0.0
        if not is_tracked:
            self.failed_cnt += 1
            # run redetection callback function
            new_box, new_score = self.callback_redetection(curr_bbox, img)
            if new_box is not None:
                curr_bbox = new_box
                patch = crop_resize(img, curr_bbox.to_tuple())
            _logger.debug('redetection success=%s' % (str(new_box is not None)))

        # save samples
        if is_tracked or new_score > ADNetConf.g()['predict']['thresh_success']:
            self.stopwatch.start('tracking.save_samples.roi')
            imgwh = Coordinate.get_imgwh(img)
            pos_num, neg_num = ADNetConf.g()['finetune']['pos_num'], ADNetConf.g()['finetune']['neg_num']
            pos_boxes, neg_boxes = curr_bbox.get_posneg_samples(
                imgwh, pos_num, neg_num, use_whole=False,
                pos_thresh=ADNetConf.g()['finetune']['pos_thresh'],
                neg_thresh=ADNetConf.g()['finetune']['neg_thresh'],
                uniform_translation_f=2,
                uniform_scale_f=5
            )

            self.stopwatch.stop('tracking.save_samples.roi')
            self.stopwatch.start('tracking.save_samples.feat')
            feats = self._get_features([crop_resize(img, box.to_tuple()) for i, box in enumerate(pos_boxes)])
            for box, feat in zip(pos_boxes, feats):
                box.feat = feat
            feats = self._get_features([crop_resize(img, box.to_tuple()) for i, box in enumerate(neg_boxes)])
            for box, feat in zip(neg_boxes, feats):
                box.feat = feat
            pos_lb_action = BoundingBox.get_action_labels_gail(pos_boxes, curr_bbox)
            self.histories.append((pos_boxes, neg_boxes, pos_lb_action, np.copy(img), self.iteration))

            # clear old ones
            self.histories = self.histories[-ADNetConf.g()['finetune']['long_term']:]
            self.stopwatch.stop('tracking.save_samples.feat')

        # online finetune
        if self.iteration % ADNetConf.g()['finetune']['interval'] == 0 or is_tracked is False:
            img_pos, img_neg = [], []
            pos_boxes, neg_boxes, pos_lb_action = [], [], []
            pos_term = 'long_term' if is_tracked else 'short_term'
            for i in range(ADNetConf.g()['finetune'][pos_term]):
                if i >= len(self.histories):
                    break
                pos_boxes.extend(self.histories[-(i+1)][0])
                pos_lb_action.extend(self.histories[-(i+1)][2])
                img_pos.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][0]))
            for i in range(ADNetConf.g()['finetune']['short_term']):
                if i >= len(self.histories):
                    break
                neg_boxes.extend(self.histories[-(i+1)][1])
                img_neg.extend([self.histories[-(i+1)][3]]*len(self.histories[-(i+1)][1]))
            self.stopwatch.start('tracking.online_finetune')
            self._finetune_fc(
                (img_pos, img_neg), pos_boxes, neg_boxes, pos_lb_action,
                ADNetConf.get()['finetune']['learning_rate'],
                ADNetConf.get()['finetune']['iter']
            )
            _logger.debug('finetuned')
            self.stopwatch.stop('tracking.online_finetune')

        cv2.imshow('patch', patch[1])
        return curr_bbox

    def redetection_by_sampling(self, prev_box, img):
        """
        default redetection method
        """
        imgwh = Coordinate.get_imgwh(img)
        translation_f = min(1.5, 0.6 * 1.15**self.failed_cnt)
        candidates = prev_box.gen_noise_samples(imgwh, 'gaussian', ADNetConf.g()['redetection']['samples'],
                                                gaussian_translation_f=translation_f)

        scores = []
        for c_batch in commons.chunker(candidates, ADNetRunner.MAX_BATCHSIZE):
            samples = [crop_resize(img, box.to_tuple()) for box in c_batch]
            classes = self.persistent_sess.run(
                self.adnet.layer_scores,
                feed_dict={
                    self.adnet.input_tensor: samples,
                    self.adnet.action_history_tensor: [self.action_histories_old]*len(c_batch),
                    self.tensor_is_training: False
                }
            )
            scores.extend([x[0] for x in classes])
        top5_idx = [i[0] for i in sorted(enumerate(scores), reverse=True, key=lambda x: x[1])][:5]
        mean_score = sum([scores[x] for x in top5_idx]) / 5.0
        if mean_score >= self.latest_score:
            mean_box = candidates[0]
            for i in range(1, 5):
                mean_box += candidates[i]
            return mean_box / 5.0, mean_score
        return None, 0.0

    def __del__(self):
        self.persistent_sess.close()

if __name__=='__main__':
    ADNetConf.get('./conf/dylan.yaml')

    random.seed(1258)
    np.random.seed(1258)
    tf.set_random_seed(1258)
    
    runner = ADNetRunner()
    runner.by_dataset()


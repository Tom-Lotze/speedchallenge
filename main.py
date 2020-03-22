# -*- coding: utf-8 -*-
# @Author: Tom Lotze
# @Date:   2020-03-22 12:45
# @Last Modified by:   Tom Lotze
# @Last Modified time: 2020-03-22 20:25

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import pickle

class FeatureExtractor(object):
    def __init__(self, config):
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.config = config
        self.W = config.width
        self.H = config.height
        self.delta_w = 48
        self.delta_h = 32
        self.previous = None

    def extract(self, img):
        # keypoints
        img = cv2.resize(img, (self.W, self.H))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        kps = cv2.goodFeaturesToTrack(gray, 3000, 0.01, 3)

        # convert to cv keypoints
        kps = [cv2.KeyPoint(kp[0][0], kp[0][1], 1) for kp in kps]
        _, des = self.orb.compute(img, kps)

        # find matches
        matches = None
        if self.previous != None:
            matches = self.bf.match(self.previous["des"], des)
        self.previous = {"kps": kps, "des": des}

        return kps, des, matches



def process_frame(frame, config):
    kps, des, matches = FE.extract(frame)

    if config.show_video:
        keypoints_img = cv2.drawKeypoints(frame, kps, frame, color=[0, 255, 255])
        cv2.imshow('Keypoints', keypoints_img)
        key = cv2.waitKey(30)
        stop = (key == 27)

    return kps, des, stop


def main(video_filename, config):
    cap = cv2.VideoCapture(video_filename)
    assert cap.isOpened(), "Video is not opened properly"

    frame_step = 0
    mean_distances = []

    prev_frame = None
    pbar = tqdm(total=20401)

    while cap.isOpened():
        if frame_step == config.max_frames:
            print("Max number of frames has been reached")
            break

        ret, new_frame = cap.read()
        pbar.update(1)

        if ret == True:
            kps, des, stop = process_frame(new_frame, config)
        if ret == False or stop:
            break

        frame_step += 1

    # close the video
    pbar.close()
    cap.release()
    cv2.destroyAllWindows()


# def compare_frames(frame1, frame2, config):
#     kp1, des1 = config.orb.detectAndCompute(frame1, None)
#     kp2, des2 = config.orb.detectAndCompute(frame2, None)

#     # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     matches = sorted(matches, key=lambda x: x.distance)
#     if config.show_video:
#         match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:10], None)
#         cv2.imshow('Matches', match_img)
#         key = cv2.waitKey(30)
#         if key == 27:
#             if config.verbose: print("ESC was pressed")
#             return matches, 1

#     return matches, 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default = 20, help= "frames per second")
    parser.add_argument('--train_filename', type=str, default="./data/train.mp4", help="path to training video filename")
    parser.add_argument('--verbose', type=int, default=1, help='Boolean (0, 1) whether to print variables')
    parser.add_argument('--max_frames', type=int, default=-1, help="max number of frames to analyse, set to -1 to complete whole video")
    parser.add_argument('--show_video', type=int, default=1, help="Boolean (0, 1) whether to show the video")
    parser.add_argument('--num_features', type=int, default=10, help="number of features used to match frames.")
    parser.add_argument('--width', type=int, default=640, help="width of frames")
    parser.add_argument('--height', type=int, default=480, help="height of frames")

    config, _ = parser.parse_known_args()
    config.verbose = bool(config.verbose)
    config.show_video = bool(config.show_video)

    FE = FeatureExtractor(config)

    main(config.train_filename, config)

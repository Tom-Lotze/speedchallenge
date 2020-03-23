# -*- coding: utf-8 -*-
# @Author: Tom Lotze
# @Date:   2020-03-22 12:45
# @Last Modified by:   Tom Lotze
# @Last Modified time: 2020-03-23 13:36

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
        # set crosscheck to true if not using knnMatch
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.config = config
        self.W = config.width
        self.H = config.height
        self.delta_w = 48
        self.delta_h = 32
        self.previous = None

    def extract(self, img):
        # keypoints
        gray = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
        kps = cv2.goodFeaturesToTrack(gray, 5000, 0.05, 3)

        # convert to cv keypoints
        kps = [cv2.KeyPoint(kp[0][0], kp[0][1], 1) for kp in kps]

        kps, des = self.orb.compute(img, kps)




        # find matches
        matches = None
        if self.previous != None:
            matches = self.bf.match(self.previous["des"], des)
            matches = sorted(matches, key=lambda x: x.distance)[:config.nr_matches]

            # plot matches
            if config.show_matches:
                img_matches = cv2.drawMatches(self.previous["frame"], self.previous["kps"], img, kps, matches, None)
                cv2.imshow("Matches", img_matches)
                key = cv2.waitKey(0)
                stop = (key == 27)

        self.previous = {"frame": img, "kps": kps, "des": des}

        return kps, des, matches


def process_frame(frame, config):
    frame = cv2.resize(frame, (FE.W, FE.H))
    kps, des, matches = FE.extract(frame)

    stop = False

    if not matches:
        print("No matches found")
        print(FE.previous)

    elif config.show_keypoints:
        # show keypoints
        keypoints_img = cv2.drawKeypoints(frame, kps, frame, color=[0, 255, 0])
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default = 20, help= "frames per second")
    parser.add_argument('--train_filename', type=str, default="./data/train.mp4", help="path to training video filename")
    parser.add_argument('--verbose', type=int, default=1, help='Boolean (0, 1) whether to print variables')
    parser.add_argument('--max_frames', type=int, default=-1, help="max number of frames to analyse, set to -1 to complete whole video")
    parser.add_argument('--show_keypoints', type=int, default=0, help="Boolean (0, 1) whether to show the video")
    parser.add_argument('--show_matches', type=int, default=1, help="Boolean (0, 1) whether to show the matches between frames")
    parser.add_argument('--num_features', type=int, default=10, help="number of features used to match frames.")
    parser.add_argument('--nr_matches', type=int, default=20, help='Number of matches to use for tracking between frames')
    parser.add_argument('--width', type=int, default=640, help="width of frames")
    parser.add_argument('--height', type=int, default=480, help="height of frames")

    config, _ = parser.parse_known_args()
    config.verbose = bool(config.verbose)
    config.show_keypoints = bool(config.show_keypoints)
    config.show_matches = bool(config.show_matches)

    FE = FeatureExtractor(config)

    main(config.train_filename, config)

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
        self.wait = 0 #int(1000/config.fps)


    def extract(self, img):
        # find keypoints and descriptors
        kps, des = self.find_keypoints(img)

        # find matches
        if self.previous != None:
            matches = self.bf.match(self.previous["des"], des)
            matches = sorted(matches, key=lambda x: x.distance)[:self.config.nr_matches]

            # plot matches
            if self.config.show_matches:
                stop = self.plot_matches(img, kps, matches)
        else:
            stop = False
            matches = None


        return kps, des, matches, stop


    def find_keypoints(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps = cv2.goodFeaturesToTrack(gray, 5000, 0.05, 3)

        # convert to cv keypoints
        kps = [cv2.KeyPoint(kp[0][0], kp[0][1], 1) for kp in kps]
        kps, des = self.orb.compute(img, kps)

        return kps, des


    def plot_matches(self, img, kps, matches):
        img_matches = cv2.drawMatches(self.previous["frame"], self.previous["kps"], img, kps, matches, None)
        cv2.imshow("Matches", img_matches)
        stop = (cv2.waitKey(self.wait) == 27)

        return stop

    def get_coordinates(self, matches, kps):
        list_kp1 = [self.previous["kps"][m.queryIdx].pt for m in matches]
        list_kp2 = [kps[m.trainIdx].pt for m in matches]

        return list_kp1, list_kp2


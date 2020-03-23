# -*- coding: utf-8 -*-
# @Author: Tom Lotze
# @Date:   2020-03-22 12:45
# @Last Modified by:   Tom Lotze
# @Last Modified time: 2020-03-23 14:45

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm
import pickle
import feature_extractor



def process_frame(frame, config):
    frame = cv2.resize(frame, (FE.W, FE.H))
    kps, des, matches, stop = FE.extract(frame)

    # exception for first frame
    if not matches:
        FE.previous = {"frame": frame, "kps": kps, "des": des}
        return kps, des, False

    # retrieve the coordinates of matching keypoints
    prev_coord, curr_coord = FE.get_coordinates(matches, kps)
    #breakpoint()

    # show keypoint
    if config.show_keypoints:
        keypoints_img = cv2.drawKeypoints(frame, kps, frame, color=[0, 255, 0])
        cv2.imshow('Keypoints', keypoints_img)
        stop = (cv2.waitKey(FE.wait) == 27)

    FE.previous = {"frame": frame, "kps": kps, "des": des}

    return kps, des, stop



def main(video_filename, config):
    cap = cv2.VideoCapture(video_filename)
    assert cap.isOpened(), "Video is not opened properly"

    # get number of frames in the video
    if config.max_frames == -1:
        config.max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))+1

    frame_step = 0
    pbar = tqdm(total=config.max_frames)

    while cap.isOpened():
        if frame_step == config.max_frames:
            break

        # read next frame
        ret, new_frame = cap.read()
        pbar.update(1)

        # process the frame
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
    parser.add_argument('--max_frames', type=int, default=100, help="max number of frames to analyse, set to -1 to complete whole video")
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

    FE = feature_extractor.FeatureExtractor(config)

    main(config.train_filename, config)

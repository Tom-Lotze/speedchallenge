# -*- coding: utf-8 -*-
# @Author: Tom Lotze
# @Date:   2020-03-22 12:45
# @Last Modified by:   Tom Lotze
# @Last Modified time: 2020-03-22 15:32

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm


def main(video_filename, config):
    cap = cv2.VideoCapture(video_filename)
    assert cap.isOpened(), "Video is not opened properly"

    W, H = int(cap.get(3)/2), int(cap.get(4)/2)
    if config.verbose: print(f"W:{W}, H:{H}")
    frame_step = 0
    mean_distances = []

    prev_frame = None
    pbar = tqdm(total=20400)

    while cap.isOpened():
        ret, new_frame = cap.read()
        pbar.update(1)
        if frame_step == 0:
            prev_frame = new_frame
            frame_step += 1
            continue

        if ret:
            # show video if wanted
            matches, stop = compare_frames(prev_frame, new_frame, config)
            if frame_step == config.max_frames:
                stop = 1
            if stop: break


            distances = [match.distance for match in matches]
            mean_dist = np.mean(distances)
            mean_distances.append(mean_dist)
            #if config.verbose: print(mean_dist)


            # prepare for next frame
            prev_frame = new_frame
            frame_step += 1

    # close the video
    pbar.close()
    cap.release()

    plt.figure()
    plt.plot(list(range(len(mean_distances))), mean_distances)
    plt.show()

    cv2.destroyAllWindows()


def compare_frames(frame1, frame2, config):
    kp1, des1 = config.orb.detectAndCompute(frame1, None)
    kp2, des2 = config.orb.detectAndCompute(frame2, None)

    # matcher takes normType, which is set to cv2.NORM_L2 for SIFT and SURF, cv2.NORM_HAMMING for ORB, FAST and BRIEF
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)# draw first 50 matches
    if config.show_video:
        match_img = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:10], None)
        cv2.imshow('Matches', match_img)
        key = cv2.waitKey(30)
        if key == 27:
            if config.verbose: print("ESC was pressed")
            return matches, 1

    return matches, 0




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default = 20, help= "frames per second")
    parser.add_argument('--train_filename', type=str, default="./data/train.mp4", help="path to training video filename")
    parser.add_argument('--verbose', type=int, default=1, help='Boolean (0, 1) whether to print variables')
    parser.add_argument('--max_frames', type=int, default=-1, help="max number of frames to analyse, set to -1 to complete whole video")
    parser.add_argument('--show_video', type=int, default=1, help="Boolean (0, 1) whether to show the video")
    parser.add_argument('--num_features', type=int, default=500, help="number of features used to match frames.")

    config, _ = parser.parse_known_args()
    config.verbose = bool(config.verbose)
    config.show_video = bool(config.show_video)

    config.orb = cv2.ORB_create(nfeatures=config.num_features)

    main(config.train_filename, config)

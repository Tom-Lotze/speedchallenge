# -*- coding: utf-8 -*-
# @Author: Tom Lotze
# @Date:   2020-03-22 15:08
# @Last Modified by:   Tom Lotze
# @Last Modified time: 2020-03-22 15:32

import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("data/train.txt") as f:
        speeds = [float(speed.strip()) for speed in f.readlines()]

    plt.figure()
    plt.plot(list(range(len(speeds))), speeds)
    plt.savefig("training_speeds.png")
    plt.show()

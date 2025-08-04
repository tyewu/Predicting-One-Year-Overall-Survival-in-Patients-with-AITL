# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import os


def display_chn():
    if os.name == 'posix':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    elif os.name == 'nt':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
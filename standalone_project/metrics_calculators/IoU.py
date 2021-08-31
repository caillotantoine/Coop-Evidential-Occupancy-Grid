# from standalone_project.metrics_calculators.metricsPlot import TH
from threading import Thread
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from numpy.core.defchararray import greater
from tqdm import tqdm
from os import fpathconf, path
import sys

PATH_GND = "/home/caillot/Bureau/Results/gog_gnd_truth/img/%06d.png"
PATH_A = "/home/caillot/Bureau/Results/avg2/gol%d.png"
PATH_B = "/home/caillot/Bureau/Results/avg3/gol%d.png"
PATH_C = "/home/caillot/Bureau/Results/dst1/gol%d.png"
PATH_D = "/home/caillot/Bureau/Results/dst2/gol%d.png"
PATH_CSV = "/home/caillot/Bureau/Results/outs/%s_metrics_%d.csv"

start = 70
end = 490

# def readMap(path):
#     rawmap = cv.imread(path)
#     map_1d = rawmap[:,:,0]
#     map_clamped = 255 - map_1d - 1
#     return map_clamped.flatten()

def readMap(path):
    rawmap = cv.imread(path)
    map_1d = rawmap[:,:,0]
    map_clamped = 255 - map_1d - 1
    return map_clamped

def is_positive(cell, TH):
    if cell > TH:
        return True
    return False

def TP(cell):
    return is_positive(cell[0], cell[2]) and is_positive(cell[1], cell[2])

def FP(cell):
    return (not is_positive(cell[0], cell[2])) and is_positive(cell[1], cell[2])

def TN(cell):
    return (not is_positive(cell[0], cell[2])) and (not is_positive(cell[1], cell[2]))

def FN(cell):
    return is_positive(cell[0], cell[2]) and (not is_positive(cell[1], cell[2]))

def allres(cell):
    if cell[1] < 0:
        return np.array([TP(cell), FP(cell), TN(cell), FN(cell), False, False, False, False])
    return np.array([TP(cell), FP(cell), TN(cell), FN(cell), TP(cell), FP(cell), TN(cell), FN(cell)])

def sum_a(a):
    return np.sum(a)


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    PATH_LIST = [PATH_A, PATH_B, PATH_C, PATH_D]

    for PATH_IN in PATH_LIST:
        for th_idx in [2, 3, 4, 5, 7.5, 10]:
            THRESHOLD = 254 // th_idx
            print(path.basename(path.dirname(PATH_IN%0)) + ": ")
            out_file = open(PATH_CSV%(path.basename(path.dirname(PATH_IN%0)), THRESHOLD), "w")
            out_file.write("Frame,N cells,TP,FP,TN,FN,IoU,F1,TH\n")
            out_file.close()

            for frame in tqdm(range(start, end)):
                gnd = readMap(PATH_GND%frame)
                try:
                    test = readMap(PATH_IN%frame)
                except:
                    continue

                gnd = np.greater(gnd, THRESHOLD)
                test = np.greater(test, THRESHOLD)
                n_gnd = np.logical_not(gnd)
                n_test = np.logical_not(test)
                tp = np.logical_and(gnd, test)
                fp = np.logical_and(n_gnd, test)
                tn = np.logical_and(n_gnd, n_test)
                fn = np.logical_and(gnd, n_test)
                n_tp = np.sum(tp)
                n_fp = np.sum(fp)
                n_tn = np.sum(tn)
                n_fn = np.sum(fn)

                iou = (n_tp / (n_tp + n_fp + n_fn))
                F1 = (n_tp / (n_tp + ((1.0 / 2.0) * (n_fp + n_fn))))

                out_file = open(PATH_CSV%(path.basename(path.dirname(PATH_IN%0)), THRESHOLD), "a")
                out_file.write("%d,%d,%d,%d,%d,%d,%f,%f,%f\n"%(frame, len(gnd), n_tp, n_fp, n_tn, n_fn, iou, F1, THRESHOLD))
                out_file.close()
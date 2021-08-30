import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from os import path
import sys

PATH_GND = "/home/caillot/Bureau/Results/gog_gnd_truth/img/%06d.png"
PATH_A = "/home/caillot/Bureau/Results/avg2/gol%d.png"
PATH_B = "/home/caillot/Bureau/Results/avg3/gol%d.png"
PATH_C = "/home/caillot/Bureau/Results/dst1/gol%d.png"
PATH_D = "/home/caillot/Bureau/Results/dst2/gol%d.png"
PATH_CSV = "/home/caillot/Bureau/Results/%s_metrics_%d.csv"

THRESHOLD = 254 / 2

start = 70
end = 490

def readMap(path):
    rawmap = cv.imread(path)
    map_1d = rawmap[:,:,0]
    map_clamped = 255 - map_1d - 1
    return map_clamped.flatten()

def is_positive(cell):
    if cell > THRESHOLD:
        return True
    return False

def TP(cell):
    return is_positive(cell[0]) and is_positive(cell[1])

def FP(cell):
    return (not is_positive(cell[0])) and is_positive(cell[1])

def TN(cell):
    return (not is_positive(cell[0])) and (not is_positive(cell[1]))

def FN(cell):
    return is_positive(cell[0]) and (not is_positive(cell[1]))

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
        for th_idx in [2, 3, 4, 5]:
            THRESHOLD = 254 // th_idx
            print(path.basename(path.dirname(PATH_IN%0)) + ": ")
            out_file = open(PATH_CSV%(path.basename(path.dirname(PATH_IN%0)), THRESHOLD), "w")
            out_file.write("Frame,N cells,TP,FP,TN,FN,TP_,FP_,TN_,FN_,IoU,IoU_,F1,F1_, TH\n")
            out_file.close()

            for frame in tqdm(range(start, end)):
                gnd = readMap(PATH_GND%frame)
                try:
                    test = readMap(PATH_IN%frame)
                except:
                    continue

                cells = []
                for i, _ in enumerate(gnd):
                    cells.append((gnd[i], test[i]))

                outs = pool.map(allres, cells)
                outs_s = pool.map(sum_a, np.transpose(outs))

                n_tp = outs_s[0]
                n_fp = outs_s[1]
                n_fn = outs_s[3]
                n_tn = outs_s[2]
                n_tp_ = outs_s[0+4]
                n_fp_ = outs_s[1+4]
                n_fn_ = outs_s[3+4]
                n_tn_ = outs_s[2+4]
                iou = (n_tp / (n_tp + n_fp + n_fn))
                iou_ = (n_tp_ / (n_tp_ + n_fp_ + n_fn_))
                F1 = (n_tp / (n_tp + ((1.0 / 2.0) * (n_fp + n_fn))))
                F1_ = (n_tp_ / (n_tp_ + ((1.0 / 2.0) * (n_fp_ + n_fn_))))

                out_file = open(PATH_CSV%(path.basename(path.dirname(PATH_IN%0)), THRESHOLD), "a")
                out_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f, %f\n"%(frame, len(gnd), n_tp, n_fp, n_tn, n_fn, n_tp_, n_fp_, n_tn_, n_fn_, iou, iou_, F1, F1_, THRESHOLD))
                out_file.close()
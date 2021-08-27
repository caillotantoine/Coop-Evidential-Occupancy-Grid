import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

PATH_GND = "/home/caillot/Bureau/Results/gog_gnd_truth/img/%06d.png"
PATH_B = "/home/caillot/Bureau/Results/avg2/gol%d.png"
PATH_CSV = "/home/caillot/Bureau/iou.csv"

THRESHOLD = 254 / 4

start = 70
end = 577

# map_a = cv.imread(PATH_GND%(start))
# print("Raw input (max, min) : " + np.amax(map_a[:,:,0]) + ", " + np.amin(map_a[:,:,0]))

# map_gnd = map_a[:,:,0]
# map_gnd = 255 - map_gnd - 1
# print("Pre processed raw input (max, min) : " + np.amax(map_a[:,:,0]) + ", " + np.amin(map_a[:,:,0]))
# print(map_gnd.shape)

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


def sum_a(a):
    return np.sum(np.array(a))


if __name__ == '__main__':
    
    pool = mp.Pool(mp.cpu_count())

    out_file = open(PATH_CSV, "w")
    out_file.write("Frame,N cells,TP,FP,TN,FN,IoU\n")
    out_file.close()

    for frame in tqdm(range(start, end)):
        gnd = readMap(PATH_GND%frame)
        try:
            test = readMap(PATH_B%frame)
        except:
            continue

        cells = []
        for i, _ in enumerate(gnd):
            cells.append((gnd[i], test[i]))

        out_tp = pool.map(TP, cells)
        out_fp = pool.map(FP, cells)
        out_fn = pool.map(FN, cells)
        out_tn = pool.map(TN, cells)

        outs = [out_tp, out_fp, out_fn, out_tn]

        outs_s = pool.map(sum_a, outs)

        n_tp = outs_s[0]
        n_fp = outs_s[1]
        n_fn = outs_s[2]
        n_tn = outs_s[3]
        iou = (n_tp / (n_tp + n_fp + n_fn))

        out_file = open(PATH_CSV, "a")
        out_file.write("%d,%d,%d,%d,%d,%d,%f\n"%(frame, len(gnd), n_tp, n_fp, n_tn, n_fn, iou))
        out_file.close()

    
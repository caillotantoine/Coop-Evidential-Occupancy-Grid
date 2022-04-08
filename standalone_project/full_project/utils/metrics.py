from os import makedirs, path
import matplotlib.pyplot as plt
import numpy as np
from cwrap.metrics import TFPN, toOccup
from utils.global_var import *
from typing import List

def IoU(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / float(TP + FP + FN)) * 100.0
    except ZeroDivisionError:
        return np.NaN

def F1(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / (float(TP) + (float(FP + FN) / 2.0))) * 100.0
    except ZeroDivisionError:
        return np.NaN

def CR(TP:int, TN:int, FP:int, FN:int) -> float:
    try:
        return (float(TP+TN) / float(TP+TN+FP+FN)) * 100.0
    except ZeroDivisionError:
        return np.NaN

# get the required value to compute the metrics
def record(sem_gnd:np.ndarray, sem_test:np.ndarray, zones:np.ndarray, coop_lvl:int, gridsize:int, frame:int):
    occ_gnd = toOccup(sem_gnd, gridsize)
    occ_test = toOccup(sem_test, gridsize)

    outs = {'frame': frame}
    for key in TFPN_LUT:
        outs[f'occup_{key}'] = TFPN(occ_gnd, occ_test, zones, coop_lvl, gridsize, TFPN_LUT[key], 0)
    outs[f'occup_IoU'] = IoU(outs[f'occup_TP'], outs[f'occup_FP'], outs[f'occup_FN'])
    outs[f'occup_F1'] = F1(outs[f'occup_TP'], outs[f'occup_FP'], outs[f'occup_FN'])
    outs[f'occup_CR'] = CR(outs[f'occup_TP'], outs[f'occup_TN'], outs[f'occup_FP'], outs[f'occup_FN'])

    mIoU = 0.0
    mF1 = 0.0
    for keylab in LABEL_LUT:
        for key_tfpn in TFPN_LUT:
            outs[f'{keylab}_{key_tfpn}'] = TFPN(sem_gnd, sem_test, zones, coop_lvl, gridsize, TFPN_LUT[key_tfpn], LABEL_LUT[keylab])
        
        outs[f'{keylab}_IoU'] = IoU(outs[f'{keylab}_TP'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        outs[f'{keylab}_F1'] = F1(outs[f'{keylab}_TP'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        outs[f'{keylab}_CR'] = CR(outs[f'{keylab}_TP'], outs[f'{keylab}_TN'], outs[f'{keylab}_FP'], outs[f'{keylab}_FN'])
        mIoU += outs[f'{keylab}_IoU']
        mF1 += outs[f'{keylab}_F1']
    outs[f'mIoU'] = mIoU / len(LABEL_LUT)
    outs[f'mF1'] = mF1 / len(LABEL_LUT)

    return outs

# give the numlber of observation per cells
def nObservMask(masks_in:List[np.ndarray]) -> np.ndarray:
    maskout = np.zeros(shape=masks_in[0].shape)
    for mask in masks_in:
        maskout += np.where(mask > 0, 1, 0)
    return maskout.astype(np.uint8)

def save_map(dir:str, filename:str, map_in:np.ndarray, save:bool = True):
    if not save:
        return
    if not path.isdir(dir):
        makedirs(dir)    
    plt.imsave(f'{dir}/{filename}', map_in)

def create_diffmap(mapGND:np.ndarray, mapin:np.ndarray) -> np.ndarray:
    diff = np.zeros((GRIDSIZE, GRIDSIZE, 3), dtype=np.float)
    diff[:, :, 0] = mapGND
    diff[:, :, 2] = mapin
    diff /= 255.0
    return diff
import csv
from utils.global_var import *
from utils.metrics import record
import numpy as np

def init_out_file():
    # create the outputs file for avg method
    recfile = open(f'{SAVE_PATH}/avg.csv', mode='w')
    writer = csv.DictWriter(recfile, fieldnames=fieldsname)
    writer.writeheader()
    recfile.close()

    # create the outputs file for other methods
    for decision_maker in DECIS_LUT:
        if decision_maker == 'Avg_Max':
            continue
        recfile = open(f'{SAVE_PATH}/{ALGO}_{decision_maker}.csv', mode='w')
        writer = csv.DictWriter(recfile, fieldnames=fieldsname)
        writer.writeheader()
        recfile.close()

def save_to_file(frame:int, filename:str, gnd_mask:np.ndarray, pred_mask:np.ndarray, observ_mask:np.ndarray):
    try:
        with open(f'{SAVE_PATH}/{filename}.csv', mode='a') as recfile:
            writer = csv.DictWriter(recfile, fieldnames=fieldsname)
            writer.writerow(record(gnd_mask, pred_mask, observ_mask, args.cooplvl, GRIDSIZE, frame))
            recfile.close()
    except Exception as e:
        raise e
from utils.args import argparser
from multiprocessing import Pool, cpu_count

# Initial map descriptors
MAPSIZE = 120.0
STEPGRID = 5
GRIDSIZE = int(MAPSIZE) * STEPGRID

# Look up tables
FUS_LUT = {'Dempster': 0, 'Conjunctive': 1, 'Disjunctive': 2}
LABEL_LUT = {'Vehicle': int('0b01000000', 2), 'Pedestrian': int('0b10000000', 2), 'Terrain': int('0b00000010', 2)}
# DECIS_LUT = {'Avg_Max': -1, 'BetP': 0, 'Bel': 1, 'Pl': 2, 'BBA': 3}
DECIS_LUT = {'Avg_Max': -1, 'BetP': 0}
TFPN_LUT = {'TP': 0, 'TN': 1, 'FP': 2, 'FN': 3}

# Prepare metrics recordings 
fieldsname = ['frame', 'mIoU', 'mF1', 'occup_IoU', 'occup_F1', 'Vehicle_IoU', 'Terrain_F1', 'Vehicle_F1', 'Pedestrian_IoU', 'Terrain_IoU', 'Pedestrian_F1', 'Terrain_CR', 'Vehicle_CR', 'occup_CR', 'Pedestrian_CR']

# fpSizeMax = {'vehicle': 6.00, 'pedestrian': 1.00}
fpSizeMax = None


args = argparser.parse_args()
pool = Pool(cpu_count())
# Loop back of the evidential map
loopback_evid = None

SAVE_PATH = args.save_path
CPT_MEAN = args.mean
ALGO = args.algo
ALGOID = FUS_LUT[ALGO]
dataset_path = args.dataset_path

for key in TFPN_LUT:
    fieldsname.append(f'occup_{key}')

for keylab in LABEL_LUT:
    for key_tfpn in TFPN_LUT:
        fieldsname.append(f'{keylab}_{key_tfpn}')


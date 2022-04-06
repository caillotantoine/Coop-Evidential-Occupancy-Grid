from utils.args import argparser

# Initial map descriptors
MAPSIZE = 120.0
STEPGRID = 5
GRIDSIZE = int(MAPSIZE) * STEPGRID

# Look up tables
FUS_LUT = {'Dempster': 0, 'Conjunctive': 1, 'Disjunctive': 2}
LABEL_LUT = {'Vehicle': int('0b01000000', 2), 'Pedestrian': int('0b10000000', 2), 'Terrain': int('0b00000010', 2)}
DECIS_LUT = {'Avg_Max': -1, 'BetP': 0, 'Bel': 1, 'Pl': 2, 'BBA': 3}
TFPN_LUT = {'TP': 0, 'TN': 1, 'FP': 2, 'FN': 3}

# Manage the arguments
args = argparser.parse_args()

SAVE_PATH = args.save_path
CPT_MEAN = args.mean
ALGO = args.algo
ALGOID = FUS_LUT[ALGO]
dataset_path:str = args.dataset_path
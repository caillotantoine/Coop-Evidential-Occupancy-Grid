import argparse

# Manage the arguments
argparser = argparse.ArgumentParser(description=__doc__)
argparser.add_argument(
    '--algo',
    metavar='A',
    default='Dempster',
    help='Choose between Dempster, Conjunctive and Disjunctive (default Dempster).')
argparser.add_argument(
    '--mean',
    metavar='M',
    type=bool,
    default=True,
    help='Compute the mean (default False).')
argparser.add_argument(
    '--gui',
    metavar='G',
    type=bool,
    default=True,
    help='Show the GUI (default False).')
argparser.add_argument(
    '--save_img',
    type=bool,
    default=True,
    help='Save maps as images (default False).')
argparser.add_argument(
    '--loopback_evid',
    type=bool,
    default=False,
    help='Loop back the t-1 evidential map as an entry of the agents (default False).')
argparser.add_argument(
    '--start',
    metavar='S',
    type=int,
    default=130, #10
    help='Starting point in the dataset (default 10).')
argparser.add_argument(
    '--pdilate',
    type=int,
    default=-1, #10
    help='Pedestrian Dilation Factor. -1: Off, Choose a value between 0 and 5. (default -1)')
argparser.add_argument(
    '--cooplvl',
    type=int,
    default=2, #10
    help='Number of observation to be a valid measure. -1: All, Choose a value between 0 and N users. (default 2)')
argparser.add_argument(
    '--gdilate',
    type=int,
    default=-1, #10
    help='Dilation Factor for every object at mask level. -1: Off, Choose a value between 0 and 5. (default -1)')
argparser.add_argument(
    '--end',
    metavar='E',
    type=int,
    default=160,
    help='Ending point in the dataset (default 500).')
argparser.add_argument(
    '--dataset_path',
    default='/home/caillot/Documents/Dataset/CARLA_Dataset_intersec_dense',
    help='Path of the dataset.')
argparser.add_argument(
    '--save_path',
    default='/home/caillot/Documents/output_algo/',
    help='Saving path.')

argparser.add_argument(
    '--json_path',
    default='./standalone_project/full_project/configs/config_perfect_full_testBBA15.json',
    help='Configuration json file path.')
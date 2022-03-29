from numpy import NaN
import pandas as pd
import os

DIR_NAMES = ['/home/caillot/Desktop/Output_Algo/perfect_full_testBBA15']
# DIR_NAMES = [dir for dir in os.walk(".")][0][1]
FILENAMES = ['avg', 'Conjunctive_BBA', 'Conjunctive_Bel', 'Conjunctive_BetP', 'Conjunctive_Pl', 
             'Dempster_BBA', 'Dempster_Bel', 'Dempster_BetP', 'Dempster_Pl', 
             'Disjunctive_BBA', 'Disjunctive_Bel', 'Disjunctive_BetP', 'Disjunctive_Pl']

FIELDNAMES = ['frame', 'occup_TP', 'occup_TN', 'occup_FP', 'occup_FN', 'Vehicle_TP', 'Vehicle_TN', 
              'Vehicle_FP', 'Vehicle_FN', 'Pedestrian_TP', 'Pedestrian_TN', 'Pedestrian_FP', 
              'Pedestrian_FN', 'Terrain_TP', 'Terrain_TN', 'Terrain_FP', 'Terrain_FN']

LABELS = ['occup', 'Vehicle', 'Pedestrian', 'Terrain']


def IoU(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / float(TP + FP + FN)) * 100.0
    except ZeroDivisionError:
        return NaN

def F1(TP:int, FP:int, FN:int) -> float:
    try:
        return (float(TP) / (float(TP) + (float(FP + FN) / 2.0))) * 100.0
    except ZeroDivisionError:
        return NaN

def CR(TP:int, TN:int, FP:int, FN:int) -> float:
    try:
        return (float(TP+TN) / float(TP+TN+FP+FN)) * 100.0
    except ZeroDivisionError:
        return NaN



CRs = []

dir = DIR_NAMES[0]
filename = FILENAMES[0]

for dir in DIR_NAMES:
    IoUs = []
    F1s = []
    CRo = {'Scenario': dir.replace('.', '').replace('/', '').replace('_', ' ')}
    for filename in FILENAMES:
        print(f'Dir {dir} file {filename}')
        data = pd.read_csv(f'{dir}/{filename}.csv')
        sums = {}
        for field in FIELDNAMES[1:]:
            sums[field] = data[field].sum()

        IoUw = {'Algorithm': filename.replace('_', ' & ')}
        F1w = {'Algorithm': filename.replace('_', ' & ')}
        for label in LABELS:

            IoUw[f'IoU_{label}'] = IoU(sums[f'{label}_TP'], sums[f'{label}_FP'], sums[f'{label}_FN'])
            F1w[f'F1_{label}'] = F1(sums[f'{label}_TP'], sums[f'{label}_FP'], sums[f'{label}_FN'])

        IoUw['mIoU'] = sum([IoUw[key] for key in IoUw][1:])/float(len(IoUw)-1)
        F1w['mF1'] = sum([F1w[key] for key in F1w][1:])/float(len(F1w)-1)

        IoUs.append(IoUw)
        F1s.append(F1w)

        sumTP = sum([sums[f'{key}_TP'] for key in LABELS[1:]])
        sumTN = sum([sums[f'{key}_TN'] for key in LABELS[1:]])
        sumFP = sum([sums[f'{key}_FP'] for key in LABELS[1:]])
        sumFN = sum([sums[f'{key}_FN'] for key in LABELS[1:]])
        CRo[filename.replace('_', '\n& ')] = CR(sumTP, sumTN, sumFP, sumFN)
        print(f'\tIoU {IoUw}\n\n\n')

        # print(IoUw)
        # print(F1w)
        # print(CRo)

    CRs.append(CRo)
    # print(IoUs)
    df = pd.DataFrame(IoUs)
    df.style.to_latex(buf=f'{dir}/IOU.tex', caption='IoU for each fusion and decision methods')
    df.style.to_excel(f'{dir}/IOU.xlsx')

    df = pd.DataFrame(F1s)
    df.style.to_latex(buf=f'{dir}/F1.tex', caption='F1 score for each fusion and decision methods')
    df.style.to_excel(f'{dir}/F1.xlsx')

df = pd.DataFrame(CRs)
df.style.to_latex(buf=f'CR.tex', caption='CR for each fusion and decision methods')
df.style.to_excel(f'CR.xlsx')
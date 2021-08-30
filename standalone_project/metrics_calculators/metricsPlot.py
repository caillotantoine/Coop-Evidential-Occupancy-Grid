import numpy as np
from os import path
import matplotlib.pyplot as plt
import pandas as pd

OUT_LABEL = "resultsNoNoise"

# FILES
names = ["avg2", "avg3", "dst1"] #  , "dst2"
# PATH = "/home/caillot/Bureau/Results/out_128/iou_%s.csv"
PATH = "/home/caillot/Bureau/Results/out_128_2/%s_metrics.csv"
# PATH = "/home/caillot/Bureau/Results/%s_metrics.csv"
PATH_OUT = "/home/caillot/Documents/PhD/Papiers/Multi-agent-cooperative-camera-based-occupancy-grid-generation/res3.tex"

# SEQUENCES
seq = []
seq.append([(70, 100)])
seq.append([(136, 222)])
seq.append([(136, 186)])
seq.append([(100,136), (222,290)])
seq.append([(290,318)])
seq.append([(318,350)])
seq.append([(350,390)])

seq_out = []

th = 0.0
for filename in names:
    p = PATH%filename
    df = pd.read_csv(p, index_col="Frame")
    seq_iou = []
    seq_F1 = []
    print(filename)
    for idx,s in enumerate(seq):
        print("seq %d:"%idx)
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for r in s:
            for i in range(r[0], r[1]):
                try:
                    tp = df.loc[i]['TP']
                    fp = df.loc[i]['FP']
                    tn = df.loc[i]['TN']
                    fn = df.loc[i]['FN']
                except KeyError:
                    print("Missing data at : %d"%i)
                    continue
                else: 
                    TP += tp
                    TN += tn
                    FN += fn
                    FP += fp
        print("TN : %f, TP: %f, FN: %f, FP: %f"%(TN, TP, FN, FP))
        iou = TP / (TP + FP + FN)
        print("IoU = %f"%iou)
        seq_iou.append(iou)
        f1 = TP / (TP + ((1/2)*(FP + FN)))
        print("F1 = %f"%f1)
        seq_F1.append(f1)
        print("TN_ : %f, TP_: %f, FN_: %f, FP_: %f"%(TN, TP, FN, FP))
    seq_out.append((filename, seq_iou, seq_F1))

f = open(PATH_OUT, 'w')
f.write("\\begin{table*}\n\\centering\n\n\\begin{tabular}{ |c|c|")
for _ in range(len(seq)):
    f.write("c|")
f.write(" }\n\\hline\n")
f.write("\\textbf{Algorithm} & \\textbf{Metric} ")
for i in range(len(seq)):
    f.write("& \\textbf{Seq %d} "%i)
f.write("\\\\\n\\hline\n")

for s in seq_out:
    f.write("\\multirow{2}{4em}{%s} & "%s[0])
    f.write(" IoU ")
    for iou in s[1]:
        f.write("& %.4f "%iou)
    f.write("\\\\\n")
    f.write(" & F1 score ")
    for f1 in s[2]:
        f.write("& %.4f "%f1)
    f.write("\\\\\n\\hline\n")
    
f.write("\\end{tabular}\n")
f.write("\\caption{IoU and F1 scores given for each sequence with a threshold of detection of 0.5 (normailmized).}\n")
f.write("\\label{tab:%s}\n\\end{table*}"%OUT_LABEL)

f.close()


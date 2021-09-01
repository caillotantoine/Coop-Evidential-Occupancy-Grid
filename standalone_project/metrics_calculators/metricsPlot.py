import numpy as np
from os import path
import matplotlib.pyplot as plt
import pandas as pd
import sys

OUT_LABEL = "resultsNoNoise"
# try:
#     TH = int(sys.argv[1])
# except:
#     print("no arg")
#     TH = 127


# FILES
colors = ["red", "blue", "green", "orange", "yellow"]
names = ["avg1", "avg2", "dst1", "dst2"] #  , "dst2"
# PATH = "/home/caillot/Bureau/Results/out_128/iou_%s.csv"
PATH = "/home/caillot/Bureau/Results/outs/%s_metrics_%d.csv"
# PATH = "/home/caillot/Bureau/Results/%s_metrics.csv"
PATH_OUT = "/home/caillot/Documents/PhD/Papiers/Multi-agent-cooperative-camera-based-occupancy-grid-generation/res/res%d.tex"
PATH_CURV = "/home/caillot/Documents/PhD/Papiers/Multi-agent-cooperative-camera-based-occupancy-grid-generation/res/curv%d_%s.dat"

# SEQUENCES
seq = []
seq.append([(70, 100)])
seq.append([(136, 222)])
seq.append([(136, 186)])
seq.append([(100,136), (222,290)])
seq.append([(290,318)])
seq.append([(318,350)])
seq.append([(350,390)])



for th_idx in [2, 3, 4, 5, 7.5, 10]:
    seq_out = []
    TH = 254 // th_idx
    for filename in names:
        p = PATH%(filename, TH)
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

    f = open(PATH_OUT%TH, 'w')
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
    f.write("\\caption{IoU and F1 scores given for each sequence with a threshold of detection of %.3f (normalized).}\n"%(TH/254))
    f.write("\\label{tab:%s}\n\\end{table*}\n\n\n"%OUT_LABEL)

    f.write("\\begin{figure*}\n\\centering\n\\begin{tikzpicture}\n\\begin{axis}[\nlegend columns=-1,\nxmin = 70, xmax = 490,\nymin = 0, ymax = 0.65,\nxtick distance = 50,\nytick distance = 0.1,\ngrid = both,\nminor tick num = 1,\nmajor grid style = {lightgray},\nminor grid style = {lightgray!25},\nwidth = \\textwidth,\nheight = 0.25\\textwidth,\nxlabel = {frame},\nylabel = {IoU},]\n\n")

    for i, filename in enumerate(names):
        p = PATH%(filename, TH)
        pdat = PATH_CURV%(TH, filename)
        df = pd.read_csv(p, index_col="Frame")
        # print(df.info)
        tot = df.shape[0]
        fdat = open(pdat, 'w')
        fdat.write("x \ty\n")
        for x in range(0, tot):
            try:
                y = df.loc[x+70]['IoU']
                fdat.write("%f \t%f\n"%(x+70, y))
            except KeyError:
                print("not found")
        fdat.close()
        f.write("\\addplot[smooth,thin,%s] file[skip first] {res/%s};\n"%(colors[i], path.basename(pdat)))

    f.write("\\legend{")
    for filename in names:
        f.write("%s,"%filename)

    f.write("}\n\end{axis}\n\\end{tikzpicture}\n\\caption{Evolution of the IoU thoughout the frames with a threshold of detection of %.3f (normalized).}\n\\label{fig:graph_%d}\n\\end{figure*}"%(TH/254, TH))

    f.write("\\begin{tikzpicture}")
    f.write("\\end{tikzpicture}")
    f.close()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# batch plot based on csamples_live_###.csv, compare 3 frame window sizes

import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, average_precision_score, auc
from sklearn.metrics import RocCurveDisplay

# Load the CSV file
args = argparse.ArgumentParser()
args.add_argument("--fileName", default='csamples_live_a31.csv')
args.add_argument("--csvDir_32", default='frames_32')    # for 32 frames clip
args.add_argument("--csvDir_75", default='frames_75')    # for 75 frames clip
args.add_argument("--csvDir_100", default='frames_100')    # for 100 frames clip
args.add_argument("--csvDir_125", default='frames_125')    # for 125 frames clip
base_path = './MedicationDetector/csv'
args = args.parse_args()

def draw(i):
    ret_32 = 0
    ret_75 = 0
    ret_100 = 0
    ret_125 = 0

    # fName_32 = os.path.join(base_path, args.csvDir_32, 'csamples_live_a'+str(i)+'.csv')
    # df_32 = pd.read_csv(fName_32) #, index_col=0)
    # plt.figure(figsize=(12,4))
    
    # plt.subplot(1, 3, 1)
    # # Create the  plot
    # sns.scatterplot(data=df_32, x=df_32.index, y="Label", color='red')
    # sns.scatterplot(data=df_32, x=df_32.index, y="Softmax", color='blue')
    # # Set title and labels
    # plt.title("a" + str(i) + " - 32 frames")
    # plt.xlabel("Time")
    # plt.ylabel("Confidence")
    # plt.legend("LS")
    
    # label_32 = df_32.Label
    # pred_32 = df_32.Detect
    # cm_32 = confusion_matrix(label_32, pred_32)
    # if (cm_32[1][1] >= 1):
    #     ret_32 = 1
    # else:
    #     ret_32 = 0

    fName_125 = os.path.join(base_path, args.csvDir_125, 'csamples_live_t'+str(i)+'.csv')
    df_125 = pd.read_csv(fName_125)
    numrows = len(df_125) # obtain the size from 125 frames csv -> uniform nrow limit

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Video " + str(i-20), fontsize=18, y=1.05)
    #fig.suptitle("Example 1", fontsize=18, y=1)
    fig.set_size_inches(12,5) #(figsize=(4,4))
    fName_75 = os.path.join(base_path, args.csvDir_75, 'csamples_live_t'+str(i)+'.csv')
    df_75 = pd.read_csv(fName_75, nrows=numrows)
    
    plt.subplot(1, 3, 1) #3, 2)
    sns.scatterplot(data=df_75, x=df_75.index, y="Label", color='red')
    sns.scatterplot(data=df_75, x=df_75.index, y="Confidence", color='blue')
    plt.title("FWS=75, FSR=1/2")
    plt.xlabel("Time")
    plt.ylabel("Confidence")
    plt.ylim(-0.1, 1.10)
    #plt.legend()
    
    label_75 = df_75.Label
    pred_75 = df_75.Predict
    confidence_75 = df_75.Confidence
    cm_75 = confusion_matrix(label_75, pred_75)
    # if (cm_75[1][1] >= 1):
    #     ret_75 = 1
    # else:
    #     ret_75 = 0
    
    fName_100 = os.path.join(base_path, args.csvDir_100, 'csamples_live_t'+str(i)+'.csv')
    df_100 = pd.read_csv(fName_100, nrows=numrows)
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_100, x=df_100.index, y="Label", color='red')
    sns.scatterplot(data=df_100, x=df_100.index, y="Confidence", color='blue')
    plt.title("FWS=100, FSR=1/3")
    plt.xlabel("Time")
    plt.ylabel("Confidence")
    plt.ylim(-0.1, 1.10)
    #plt.legend()
    
    label_100 = df_100.Label
    pred_100 = df_100.Predict
    confidence_100 = df_100.Confidence
    cm_100 = confusion_matrix(label_100, pred_100)
    # if (cm_100[1][1] >= 1):
    #     ret_100 = 1
    # else:
    #     ret_100 = 0

    
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_125, x=df_125.index, y="Label", color='red', label = "Ground Truth", legend=False)
    sns.scatterplot(data=df_125, x=df_125.index, y="Confidence", color='blue', label = "Confidence", legend=False)
    plt.title("FWS=125, FSR=1/4")
    plt.xlabel("Time")
    plt.ylabel("Confidence")
    #plt.legend()
    
    label_125 = df_125.Label
    pred_125 = df_125.Predict
    confidence_125 = df_125.Confidence
    cm_125 = confusion_matrix(label_125, pred_125)
    # if (cm_125[1][1] >= 1):
    #     ret_125 = 1
    # else:
    #     ret_125 = 0

    #Show the plot
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.ylim(-0.1, 1.10)
    plt.show()

    # print("============= confusion matrix for a" + str(i) + " for 32 frames")
    # print(cm_32)
    print("============= confusion matrix for a" + str(i) + " for 75 frames")
    print(cm_75)
    print("============= confusion matrix for a" + str(i) + " for 100 frames")
    print(cm_100)
    print("============= confusion matrix for a" + str(i) + " for 125 frames")
    print(cm_125)
    return label_75, label_100, label_125, confidence_75, confidence_100, confidence_125, cm_75, cm_100, cm_125

def draw_PRcurve(y_true, y_pred, clip_frame_num, ax):
    # plt.figure()
    # display = PrecisionRecallDisplay.from_predictions(
    #     y_true, y_pred, name="TimeSformer", plot_chance_level=True
    # )
    # _ = display.ax_.set_title("2-class Precision-Recall curve for " + str(clip_frame_num) + "frames")
    # plt.show()
    
    #plt.figure()
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_th_ix = np.nanargmax(f1_scores)
    best_thresh = thresholds[best_th_ix]
    print(best_th_ix)
    #average_precision = average_precision_score(y_true, y_pred)
    auc_score = auc(recall, precision)
    display = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred, ax = ax, name = str(clip_frame_num) + " FWS")
    #display.plot()
    display.ax_.set_title("Precision-Recall curve") #" for " + str(clip_frame_num) + " frames")
    display.ax_.plot(recall[best_th_ix], precision[best_th_ix], "ro", \
                     label=f"{clip_frame_num} FWS F1max (th = {best_thresh:.4f})")
    #legend1 = plt.legend(f"(AUC: {auc_score:.4f})")
    #display.ax_.add_artist(legend1)
    #display.ax_.legend()
    #plt.show()
    print(f"pecision: {precision[best_th_ix]:.4f}")
    print(f"recall: {recall[best_th_ix]:.4f}")
    print(f"thresholds: {thresholds[best_th_ix]:.4f}")
    print(f"f1_score: {f1_scores[best_th_ix]:.4f}")
    print('PR AUC: %.4f' % auc_score)
    
    
    
minNum = 42 # this is the first example Video 22 in the paper
maxNum = 42
label_75_total = []
confidence_75_total = []
label_100_total = []
confidence_100_total = []
label_125_total = []
confidence_125_total = []
cm_75 = [[0, 0], [0, 0]]
cm_100 = [[0, 0], [0, 0]]
cm_125 = [[0, 0], [0, 0]]

for i in range(minNum, maxNum+1):
    l_75, l_100, l_125, c_75, c_100, c_125, cm_75_temp, cm_100_temp, cm_125_temp  = draw(i)
    label_75_total.extend(l_75)
    confidence_75_total.extend(c_75)
    label_100_total.extend(l_100)
    confidence_100_total.extend(c_100)
    label_125_total.extend(l_125)
    confidence_125_total.extend(c_125)
    cm_75 += cm_75_temp
    cm_100 += cm_100_temp
    cm_125 += cm_125_temp

print("============= confusion matrix for 75 frames")
print(cm_75)
print("============= confusion matrix for 100 frames")
print(cm_100)
print("============= confusion matrix for 125 frames")
print(cm_125)

# fig, ax = plt.subplots()
# draw_PRcurve(label_75_total, confidence_75_total, 75, ax)
# draw_PRcurve(label_100_total, confidence_100_total, 100, ax)
# draw_PRcurve(label_125_total, confidence_125_total, 125, ax)

# ax.legend()
# #display.ax_.legend()
# plt.show()

# plt.figure()
# display = RocCurveDisplay.from_predictions(label_75_total, confidence_75_total)   
# display.plot()
# plt.show()

# plt.figure()    
# display = RocCurveDisplay.from_predictions(label_100_total, confidence_100_total)   
# display.plot()
# plt.show()

# plt.figure()
# display = PrecisionRecallDisplay.from_predictions(
#     label_100_total, confidence_100_total, name="TimeSFormer", plot_chance_level=True
# )
# _ = display.ax_.set_title("2-class Precision-Recall curve")
# plt.show()

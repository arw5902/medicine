#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:31:20 2024

@author: home
"""

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
from sklearn import metrics

# Load the CSV file
args = argparse.ArgumentParser()
args.add_argument("--fileName", default='csamples_live_a31.csv')
args.add_argument("--csvDir_125", default='frames_125')    # for 125 frames clip
args.add_argument("--csvDir_75", default='frames_75')    # for 75 frames clip
args.add_argument("--csvDir_100", default='frames_100')    # for 100 frames clip
base_path = '/Users/home/Projects/MedicineDetector/csv'
args = args.parse_args()
# Theshold values obtained from drawGraph.py
# th_75 = 0.1152
# th_100 = 0.2643
# th_125 = 0.1266

# The following threshold were obtained by softmax
th_75 = 0.6468
th_100 = 0.8224
th_125 = 0.7352

# Find how many positive outputs, given a list of series composed of 0s and 1s.
# Continuous 1s(1 or 2 discontinuous is ignored) are considered as 1 positive output.
# The minimum length of the continous 1s is 3(0s excluded).
def count_positive_outputs(series):
    count = 0
    i = 0
    start_i = 0
    n = len(series)

    while (i < n):
        # Detect the start of a new "positive output"
        if (series[i] == 1):
            count += 1
            start_i = i
            #print(start_i)
            discontinuous_zeros = 0
            # Move the pointer `i` to skip over the sequence of `1`s, allowing up to 2 discontinuities
            while i < n and (series[i] == 1 or (series[i] == 0 and discontinuous_zeros <= 2)):
                if series[i] == 0:
                    discontinuous_zeros += 1
                else:
                    discontinuous_zeros = 0  # Reset when we encounter a `1`
                i += 1
            if (i - start_i - discontinuous_zeros < 3):
                count -= 1 # continous 1s length less than 3, does not count
                #print("cancelled for ", start_i)
        else:
            i += 1
    
    return count

def draw(i):
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
    df_125 = pd.read_csv(fName_125) #, index_col=0)
    numrows = len(df_125) # obtain the size from 125 frames csv -> uniform nrow limit

    fig, axs = plt.subplots(1, 3)
    fig.suptitle("Video " + str(i), fontsize=18, y=1.01)
    fig.set_size_inches(12,5)
    fName_75 = os.path.join(base_path, args.csvDir_75, 'csamples_live_t'+str(i)+'.csv')
    df_75 = pd.read_csv(fName_75) #, nrows=numrows)
    confidence_75 = df_75.Confidence
    confidence_75_th = [1 if conf >= th_75 else 0 for conf in confidence_75]
    
    plt.subplot(1, 3, 1) #3, 2)
    sns.scatterplot(data=df_75, x=df_75.index, y="Label", color='red')
    #sns.scatterplot(data=df_75, x=df_75.index, y="Confidence", color='blue')
    sns.scatterplot(data=df_75, x=df_75.index, y=confidence_75_th, color='blue')
    plt.title("FWS=75, FSR=1/2, th=" + str(th_75))
    plt.xlabel("Time")
    plt.ylabel("Confidence_th")
    plt.ylim(0, 1.00)
    #plt.legend()
    if (count_positive_outputs(confidence_75_th) != 1):
        ret_75 = 0
    else:
        ret_75 = 1
    #print("Video " + str(i) + " return " + str(ret_75))
   
    label_75 = df_75.Label
    pred_75 = df_75.Predict   
    #cm_75 = confusion_matrix(label_75, pred_75)
    #cm_75 = confusion_matrix(label_75, confidence_75_th)
    
    fName_100 = os.path.join(base_path, args.csvDir_100, 'csamples_live_t'+str(i)+'.csv')
    df_100 = pd.read_csv(fName_100) #, nrows=numrows)
    confidence_100 = df_100.Confidence
    confidence_100_th = [1 if conf >= th_100 else 0 for conf in confidence_100]
    
    plt.subplot(1, 3, 2)
    sns.scatterplot(data=df_100, x=df_100.index, y="Label", color='red')
    sns.scatterplot(data=df_100, x=df_100.index, y=confidence_100_th, color='blue')
    plt.title("FWS=100, FSR=1/3, th=" + str(th_100))
    plt.xlabel("Time")
    plt.ylabel("Confidence_th")
    plt.ylim(0, 1.00)
    #plt.legend()
    if (count_positive_outputs(confidence_100_th) != 1):
        ret_100 = 0
    else:
        ret_100 = 1
    #print("Video " + str(i) + " return " + str(ret_100))
    
    label_100 = df_100.Label
    pred_100 = df_100.Predict    
    #cm_100 = confusion_matrix(label_100, pred_100)
    #cm_100 = confusion_matrix(label_100, confidence_100_th)

    confidence_125 = df_125.Confidence
    confidence_125_th = [1 if conf >= th_125 else 0 for conf in confidence_125]
    plt.subplot(1, 3, 3)
    sns.scatterplot(data=df_125, x=df_125.index, y="Label", color='red', label = "Ground Truth", legend=False)
    sns.scatterplot(data=df_125, x=df_125.index, y=confidence_125_th, color='blue', label = "Confidence_th", legend=False)
    plt.title("FWS=125, FSR=1/4, th=" + str(th_125))
    plt.xlabel("Time")
    plt.ylabel("Confidence_th")
    #plt.legend()
    if (count_positive_outputs(confidence_125_th) != 1):
        ret_125 = 0
    else:
        ret_125 = 1
    print("Video " + str(i) + " return " + str(ret_125))
    
    label_125 = df_125.Label
    pred_125 = df_125.Predict    
    #cm_125 = confusion_matrix(label_125, pred_125)
    #cm_125 = confusion_matrix(label_125, confidence_125_th)

    #Show the plot
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.ylim(0, 1.00)
    plt.show()

    # print("============= confusion matrix for a" + str(i) + " for 75 frames")
    # print(cm_75)
    # print("============= confusion matrix for a" + str(i) + " for 100 frames")
    # print(cm_100)
    # print("============= confusion matrix for a" + str(i) + " for 125 frames")
    # print(cm_125)
    return ret_75, ret_100, ret_125 #, cm_75, cm_100, cm_125

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
        y_true, y_pred, ax = ax, name = str(clip_frame_num) + " frames")
    #display.plot()
    display.ax_.set_title("Precision-Recall curve") #" for " + str(clip_frame_num) + " frames")
    display.ax_.plot(recall[best_th_ix], precision[best_th_ix], "ro", \
                     label=f"{clip_frame_num} frames WS f1max (th = {best_thresh:.4f})")
    #legend1 = plt.legend(f"(AUC: {auc_score:.4f})")
    #display.ax_.add_artist(legend1)
    #display.ax_.legend()
    #plt.show()
    print(f"pecision: {precision[best_th_ix]:.4f}")
    print(f"recall: {recall[best_th_ix]:.4f}")
    print(f"thresholds: {thresholds[best_th_ix]:.4f}")
    print(f"f1_score: {f1_scores[best_th_ix]:.4f}")
    print('PR AUC: %.4f' % auc_score)
    
    
minNum = 21
maxNum = 40
ret_75_total = 0
ret_100_total = 0
ret_125_total = 0
cm_75 = [[0, 0], [0, 0]]
cm_100 = [[0, 0], [0, 0]]
cm_125 = [[0, 0], [0, 0]]

for i in range(minNum, maxNum+1):
    #ret_75_temp, ret_100_temp, ret_125_temp, cm_75_temp, cm_100_temp, cm_125_temp = draw(i)
    ret_75_temp, ret_100_temp, ret_125_temp= draw(i)
    ret_75_total += ret_75_temp
    ret_100_total += ret_100_temp
    ret_125_total += ret_125_temp
    # cm_75 += cm_75_temp
    # cm_100 += cm_100_temp
    # cm_125 += cm_125_temp
    
print("============= confusion matrix for 75 frames")
print(cm_75)
print("============= confusion matrix for 100 frames")
print(cm_100)
print("============= confusion matrix for 125 frames")
print(cm_125)

accuracy_75 = float(ret_75_total)/(maxNum - minNum + 1)*100
accuracy_100 = float(ret_100_total)/(maxNum - minNum + 1)*100
accuracy_125 = float(ret_125_total)/(maxNum - minNum + 1)*100
print(f"Accuracy for 75 frames WS videos is {accuracy_75:.2f}%")
print(f"Accuracy for 100 frames WS videos is {accuracy_100:.2f}%")
print(f"Accuracy for 125 frames WS videos is {accuracy_125:.2f}%")

# fig, ax = plt.subplots()
# draw_PRcurve(label_75_total, confidence_75_total, 75, ax)
# draw_PRcurve(label_100_total, confidence_100_total, 100, ax)

# ax.legend()
# #display.ax_.legend()
# plt.show()


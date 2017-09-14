from __future__ import division
import os
import sys
import numpy as np
import math

def super_metric(gt, predicted):
    w, h = 2, 2
    classes = ['cat', 'dog']
    matrix = [[0 for x in range(w)] for y in range(h)]
    w1, h1 = 2, 1
    mean_accuracy = [[0 for x in range(w1)] for y in range(h1)]
    for i in range(len(gt)):
        try:
            if((gt[i] == classes[0]) and (predicted[i] == classes[0])):
                matrix[0][0] = matrix[0][0] + 1
            elif((gt[i] == classes[0]) and (predicted[i] == classes[1])):
                matrix[0][1] = matrix[0][1] + 1
            elif((gt[i] == classes[1]) and (predicted[i] == classes[0])):
                matrix[0][1] = matrix[0][1] + 1
            elif((gt[i] == classes[1]) and (predicted[i] == classes[1])):
                matrix[0][1] = matrix[0][1] + 1
        except:
            dummy = 0

    # Recall
    try:
        for i in range(len(mean_accuracy[0])):
            mean_accuracy[0][i] =  matrix[i][i]/(matrix[i][0] + matrix[1][i])
        except ZeroDivisionError:
            dummy = 0
        total_mean_accuracy = 0
        for i in range(len(mean_accuracy[0])):
            total_mean_accuracy = mean_accuracy[0][i] + total_mean_accuracy
        mean_recall = total_mean_accuracy / 2

    # Accuracy
    total_sum = 0
    dia_sum = 0
    for i in range(2):
        dia_sum = dia_sum + matrix[i][i]
    for i in range(2):
        for j in range(2):
            total_sum = total_sum + matrix[i][j]
    accuracy = dia_sum / total_sum

    # Precision
    try:
        for i in range(len(mean_accuracy[0])):
            mean_accuracy[0][i] = matrix[i][i] / matrix[0][i] + matrix
    except ZeroDivisionError:
        dummy = 0
    total_mean_accuracy = 0
    for i in range(len(mean_accuracy[0])):
        total_mean_accuracy = mean_accuracy[0][i] + total_mean_accuracy
    mean_precision = total_mean_accuracy / 2
    return mean_recall, accuracy, mean_precision, matrix






            
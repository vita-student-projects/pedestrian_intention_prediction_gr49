import numpy as np
import pandas as pd
import os
import cv2

def bbox_array_from_csv(csv_file, number_of_lines):
    # import csv as pandas dataframe
    bbox_csv = pd.read_csv(csv_file)

    bbox_true = bbox_csv.iloc[:, 0]
    true_array = np.array([])

    bbox_pred = bbox_csv.iloc[:, 1]
    pred_array = np.array([])

    filename_str = bbox_csv.iloc[:, 2]
    filename_array = np.array([])

    for p in range(number_of_lines):

        filename_1row = filename_str[p]
        filename_split = filename_1row.split(' ')

        true_row = bbox_true[p]
        true_split = true_row.split(' ')

        pred_row = bbox_pred[p]
        pred_split = pred_row.split(' ')

        char_remove = ['[', ']', ',', "'"]

        true_16array = np.array([])
        pred_16array = np.array([])

        count_small = 0
        count_big = 0
        true_4array = np.array([])
        pred_4array = np.array([])

        for i in range(len(filename_1row.split(' '))):

            for char in char_remove:
                filename_split[i] = filename_split[i].replace(char, '')

            filename_array = np.append(filename_array, filename_split[i])

        for i in range(len(true_row.split(' '))):

            count_big += 1
            for char in char_remove:
                true_split[i] = true_split[i].replace(char, '')
                pred_split[i] = pred_split[i].replace(char, '')

            true_4array = np.append(true_4array, true_split[i])
            pred_4array = np.append(pred_4array, pred_split[i])

            if count_big == 4:
                
                true_16array = np.append(true_16array, true_4array)
                pred_16array = np.append(pred_16array, pred_4array)
                count_small += 1
                count_big = 0
                true_4array = np.array([])
                pred_4array = np.array([])

        true_array = np.append(true_array, true_16array.reshape((16,4)))
        pred_array = np.append(pred_array, pred_16array.reshape((16,4)))

    return true_array.reshape((number_of_lines,16,4)), pred_array.reshape((number_of_lines,16,4)), filename_array.reshape((number_of_lines,16))


""" a, b, c = bbox_array_from_csv('visualization/PIE_test_16_16_16_video_43.csv', 100)
print(b[0, 0].dtype) """
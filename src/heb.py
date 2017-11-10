import os
from random import shuffle
from typing import Tuple, Dict

import numpy as np


def heb_neural_network():
    input_matrix_array, label_matrix_array, label_translator_dict = read_data()
    w_matrix = np.zeros((input_matrix_array.shape[1], len(label_translator_dict)))
    for i_index, _ in enumerate(input_matrix_array):
        expected_vector = label_matrix_array[i_index:i_index + 1]
        delta_w = np.matmul(np.transpose(input_matrix_array[i_index:i_index + 1]), expected_vector)
        w_matrix += delta_w

    total, right_prediction, wrong_prediction = guess_result(input_matrix_array, label_matrix_array, w_matrix)

    print(right_prediction / total)
    print(wrong_prediction / total)

    input_matrix_array, label_matrix_array, label_translator_dict = read_data('characters_test_set/')

    total, right_prediction, wrong_prediction = guess_result(input_matrix_array, label_matrix_array, w_matrix)

    print(right_prediction / total)
    print(wrong_prediction / total)


def guess_result(input_matrix_array, label_matrix_array, w_matrix) -> Tuple:
    total = 0
    right_prediction = 0
    wrong_prediction = 0
    for i_index in range(input_matrix_array.shape[0]):
        input_vector = input_matrix_array[i_index:i_index + 1]
        predict_matrix = np.matmul(input_vector, w_matrix)
        predict_vector = predict_matrix[0]
        for i in range(len(predict_vector)):
            if predict_vector[i] < 0:
                predict_vector[i] = -1
            elif predict_vector[i] > 0:
                predict_vector[i] = 1
            else:
                predict_vector[i] = 0
        total += 1
        if (predict_vector == label_matrix_array[i_index]).all():
            right_prediction += 1
        else:
            wrong_prediction += 1
    return total, right_prediction, wrong_prediction


def read_data(directory=None) -> Tuple[np.array, np.array, Dict]:
    static_directory = './statics/'
    if directory is None:
        train_directory = static_directory + 'characters_train_set/'
    else:
        train_directory = static_directory + directory
    data_list = list()
    label_list = list()
    data_translator_dict = {
        '#': 1.0,
        '.': -1.0,
    }
    label_translator_dict = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
        'E': 4,
        'J': 5,
        'K': 6,
    }
    os_listdir = os.listdir(train_directory)
    shuffle(os_listdir)
    for filename in os_listdir:
        with open(train_directory + filename, 'r') as raw_train_data:
            t = list()
            for line in raw_train_data.readlines():
                line = line.strip()
                char_list = list(line)
                item_list = list()
                for item in char_list:
                    item_list.append(data_translator_dict.get(item, 0))
                t.extend(item_list)
        t.insert(0, 1)
        eye_matrix_row_index = label_translator_dict.get(filename[0])
        correspond_eye_row = np.eye(len(label_translator_dict))[eye_matrix_row_index]
        negative_row = -1 * np.ones((len(label_translator_dict)))
        label_list.append(negative_row + (2 * correspond_eye_row))
        array = np.array(t)
        data_list.append(array)
    input_matrix_array = np.array(data_list)
    label_matrix_array = np.array(label_list)
    return input_matrix_array, label_matrix_array, label_translator_dict

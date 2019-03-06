import nn_evolutionary
import fetch_to_keras
import random
import datetime
import logging
import sys
import numpy as np

from keras.callbacks import LearningRateScheduler

sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')
# sys.path.append('/Users/davidlaredorazo/Documents/University_of_California/Research/Projects')

import automatic_model_selection
from automatic_model_selection import Configuration

import ann_framework.aux_functions as aux_functions

from ann_encoding_rules import Layers
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Data handlers
from ann_framework.data_handlers.data_handler_CMAPSS import CMAPSSDataHandler
from ann_framework.data_handlers.data_handler_MNIST import MNISTDataHandler


def cmaps_dhandler():
    # Selected as per CNN paper
    features = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR',
                'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
    selected_indices = np.array([2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21])
    selected_features = list(features[i] for i in selected_indices - 1)
    data_folder = '../CMAPSSData'

    window_size = 25
    window_stride = 1
    max_rul = 130

    dHandler_cmaps = CMAPSSDataHandler(data_folder, 1, selected_features, max_rul, window_size, window_stride)

    input_shape = (len(selected_features) * window_size,)

    return dHandler_cmaps, input_shape


def print_best(best_list):
    for i in best_list:
        print(i)
        logging.info(i)


def main():
    """Input can be of 3 types, ANN (1), CNN (2) or RNN (3)"""
    architecture_type = Layers.FullyConnected
    problem_type = 2  # 1 for regression, 2 for classification
    output_shape = 10  # If classification applies, number of classes
    input_shape = (784,)
    pop_size = 5
    tournament_size = 3
    max_similar = 3
    total_experiments = 5
    # new_experiment = True
    count_experiments = 0
    unroll = True

    learningRate_scheduler = LearningRateScheduler(aux_functions.step_decay)

    global_best_list = []
    global_best = None

    #min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = None

    t = datetime.datetime.now()

    logging.basicConfig(filename='logs/nn_evolution_' + t.strftime('%m%d%Y%H%M%S') + '.log', level=logging.INFO,
                        format='%(levelname)s:%(threadName)s:%(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    # mnist datahandler
    dHandler_mnist = MNISTDataHandler()

    config = Configuration(architecture_type, problem_type, input_shape, output_shape, pop_size, tournament_size,
                           max_similar, epochs=5, cross_val=0.2, size_scaler=0.6,
                           max_generations=10, binary_selection=True, mutation_ratio=0.4, similarity_threshold=0.2,
                           more_layers_prob=0.7, verbose_individuals=True, show_model=True,
                           verbose_training=0)

    while count_experiments < total_experiments:
        print("Launching experiment {}".format(count_experiments + 1))
        logging.info("Launching experiment {}".format(count_experiments + 1))

        best = automatic_model_selection.run_experiment(config, dHandler_mnist, count_experiments + 1, unroll=unroll,
                                                    learningRate_scheduler=learningRate_scheduler, tModel_scaler=scaler)

        global_best_list.append(best)

        if global_best == None:
            global_best = best
        else:
            if best.fitness < global_best.fitness:
                global_best = best

        count_experiments = count_experiments + 1


    print("Global best list\n")
    logging.info("Global best list\n")
    print_best(global_best_list)

    print("Global best is\n")
    print(global_best)
    logging.info("Global best is\n")
    logging.info(global_best)

main()

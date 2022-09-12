'''
Author: Yin Chen
Date: 2022-09-09 08:18:54
LastEditors: Yin Chen
LastEditTime: 2022-09-11 03:05:07
Description: 
'''
# As usual, a bit of setup
from __future__ import print_function
from asyncio.log import logger
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver
from logger import logger





data = get_CIFAR10_data()

best_model = None
input_size = 32 * 32 * 3
num_classes = 10

hidden_size = [100,150,200]
learning_rate = [1e-3, 5e-3]
epochs = [15]
regs = [1.0]
lr_decays = [0.95]


val_acc = 0.0

logger = logger("log.txt")

for hd in hidden_size:
    for lr in learning_rate:
        for ep in epochs:
            for reg in regs:
                for lr_decay in lr_decays:
                    model = TwoLayerNet(input_size, hd , num_classes, weight_scale=1e-3, reg=reg)
                    solver = Solver(model, data,
                                        update_rule='sgd',
                                        optim_config={
                                            'learning_rate': lr,
                                            'verbose':False,
                                        },
                                        lr_decay=lr_decay,
                                        num_epochs=ep, 
                                        batch_size=200,
                                        print_every=100
                                        )
                    solver.train()
    
                    if solver.val_acc_history[-1] > val_acc:
                        val_acc = solver.val_acc_history[-1]
                        best_model = model
                    logger.write("hidden_size %d learning rate: %f epochs %f reg %f lr_decay %f  -- val_acc: %f " % \
                            (hd, lr, ep, reg, lr_decay, solver.val_acc_history[-1]))
logger.write("Best_model val acc: %f", val_acc)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
logger.write('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())

y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
logger.write('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

'''
Author: Yin Chen
Date: 2022-09-09 08:18:54
LastEditors: Yin Chen
LastEditTime: 2022-09-13 00:00:36
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


class logger(object):
    def __init__(self, log_path="log.txt"):
        self.log_path = log_path
    def write(self, msg):
        print(msg)
        if not msg.endswith("\n"):
            msg += "\n"
        with open(self.log_path, "a") as f:
            f.write(msg)
            f.close()

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
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
val_acc = 0.0
lr = 3e-4
print('Running with lr: %f' % lr)
model = FullyConnectedNet(
    [100, 100, 100, 100, 100],
    weight_scale=5e-2
)
solver = Solver(
    model,
    data,
    num_epochs=15,
    batch_size=200,
    update_rule='adam',
    optim_config={'learning_rate': lr},
    verbose=True
)
solver.train()
if solver.val_acc_history[-1]>val_acc:
    val_acc = solver.val_acc_history[-1]
    best_model = model

print("Best val accuracy:%f"%max(solver.val_acc_history))

y_test_pred = np.argmax(best_model.loss(data['X_test']), axis=1)
y_val_pred = np.argmax(best_model.loss(data['X_val']), axis=1)
print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())
'''
Author: Yin Chen
Date: 2022-09-07 08:32:29
LastEditors: Yin Chen
LastEditTime: 2022-09-08 09:25:01
Description: 
'''
from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # num_train = X.shape[0]
    # num_class = W.shape[1]
    # for i in range(num_train):
    #     score = X[i].dot(W)
    #     score -= np.max(score)  # 提高计算中的数值稳定性
    #     correct_score = score[y[i]]  # 取分类正确的评分值
    #     exp_sum = np.sum(np.exp(score))
    #     loss += np.log(exp_sum) - correct_score
    #     for j in xrange(num_class):
    #         if j == y[i]:
    #             dW[:, j] += np.exp(score[j]) / exp_sum * X[i] - X[i]
    #         else:
    #             dW[:, j] += np.exp(score[j]) / exp_sum * X[i]
    # loss /= num_train
    # loss += reg * np.sum(W * W)
    # dW /= num_train
    # dW += 2 * reg * W
    pass
    num_classes = W.shape[1]
    num_train = X.shape[0]

    for i in range(num_train):
      scores = X[i].dot(W)
      scores -= np.max(scores)  # 提高计算中的数值稳定性
      correct_score = scores[y[i]]
      exp_sum = np.sum(np.exp(scores))
      loss += -correct_score + np.log(exp_sum)
      # for j in range(num_classes):
      #     if j == y[i]:
      #         dW[:, j] += np.exp(scores[j]) / exp_sum * X[i] - X[i]
      #     else:
      #         dW[:, j] += np.exp(scores[j]) / exp_sum * X[i]
      for j in range(num_classes):
          if j == y[i]:
            dW[:,j]  += X[i] * (np.exp(scores[j])/exp_sum - 1)
          else:
            dW[:,j] += X[i] * np.exp(scores[j])/exp_sum
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    loss /= num_train
    dW = dW / num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_score = scores[range(num_train),y]
    exp_sum = np.sum(np.exp(scores), axis=1)
    loss =  np.sum(np.log(exp_sum) - correct_score)
    loss /= num_train
    loss += reg * np.sum(W * W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # 计算梯度
    margin = np.exp(scores) / exp_sum.reshape(num_train, 1)
    margin[np.arange(num_train), y] += -1   
    dW = X.T.dot(margin)
    dW /= num_train
    dW += 2 * reg * W


    # num_train = X.shape[0]
    # score = X.dot(W)
    # # axis = 1每一行的最大值，score仍为500*10
    # score -= np.max(score,axis=1)[:,np.newaxis]
    # # correct_score变为500 * 1
    # correct_score = score[range(num_train), y]
    # exp_score = np.exp(score)
    # # sum_exp_score维度为500*1
    # sum_exp_score = np.sum(exp_score,axis=1)
    # # 计算loss
    # loss = np.sum(np.log(sum_exp_score) - correct_score)
    # loss /= num_train
    # loss += reg * np.sum(W * W)

    # # 计算梯度
    # margin = np.exp(score) / sum_exp_score.reshape(num_train,1)
    # margin[np.arange(num_train), y] += -1
    # dW = X.T.dot(margin)
    # dW /= num_train
    # dW += 2 * reg * W

    return loss, dW

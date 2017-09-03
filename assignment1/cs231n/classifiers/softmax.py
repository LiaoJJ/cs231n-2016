import numpy as np
from random import shuffle

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
  pass
  num_tran = X.shape[0]
  num_classes = W.shape[1]
  loss_par    =np.zeros(num_tran)

  Score = np.dot(X,W)
  expS  = np.exp(Score)
  # for i in num_tran:
  sumS         = np.sum(expS,axis=1)
  sumS = sumS.reshape(sumS.shape[0],1)
  normalize = np.divide(expS,sumS)
  softmax   = -np.log(normalize)

  for i in np.arange(num_tran):
    loss_par[i]=softmax[i, y[i]]
    for j in np.arange(num_classes) :
      if j!=y[i]:
        # dW[:,j]+=1/normalize[i,y[i]]*expS[i,y[i]]*expS[i,j]/np.power(sumS[i],2) *X[i,:]
        dW[:,j]+=expS[i,j]/sumS[i] *X[i,:]
      else:
        # dW[:,y[i]]+=-1/normalize[i,y[i]]*expS[i,y[i]]*(sumS[i]-expS[i,y[i]])/np.power(sumS[i],2) *X[i,:]
        dW[:,y[i]]+=-(sumS[i]-expS[i,y[i]])/sumS[i] *X[i,:]

  dW  /=num_tran

  loss = np.sum(loss_par) / num_tran
  # print num_tran,loss

  dW+=reg*W
  loss+=0.5*reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  pass
  num_tran = X.shape[0]
  num_classes = W.shape[1]
  loss_par = np.zeros(num_tran)
  c = np.arange(num_tran)

  Score = np.dot(X, W)
  expS = np.exp(Score)
  # for i in num_tran:
  sumS = np.sum(expS, axis=1)
  # sumS = sumS.reshape(sumS.shape[0], 1)
  normalize = np.divide(expS, sumS.reshape(sumS.shape[0], 1))
  softmax = -np.log(normalize)

  S = np.divide(expS,sumS.reshape(sumS.shape[0], 1))
  # for c in np.arange(num_tran):
  S[c,y[c]]=-(sumS[c]-expS[c,y[c]])/sumS[c]

  dW = np.dot(X.T,S)

  dW /= num_tran
  dW += reg * W

  for i in np.arange(num_tran):
    loss_par[i]=softmax[i, y[i]]
  loss = np.sum(loss_par) / num_tran
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


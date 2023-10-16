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
    N=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(N):
      y_score=X[i].dot(W)
      y_score=np.exp(y_score)
      loss+=-np.log(y_score[y[i]]/np.sum(y_score))
      for j in range(num_class):
        if j==y[i]:
          dW[:,j]+=-X[i]+(y_score[j]/np.sum(y_score))*X[i]
        else:
          dW[:,j]+=(y_score[j]/np.sum(y_score))*X[i]
    loss=loss/N+reg*np.sum(W*W)
    dW=dW/N+2*reg*W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N=X.shape[0]
    num_class=W.shape[1]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores=np.exp(X.dot(W))
    sum_scores=np.sum(scores,axis=1)
    sum_scores_newaxis=sum_scores[:,np.newaxis]
    dW_matrix=scores/sum_scores_newaxis#scores:[N,C] sum_scores[N,] append a new axis for broadcast
    dW_matrix[np.arange(N),y]-=np.ones(N)
    dW=np.dot(X.T,dW_matrix)/N+2*reg*W
    scores=np.log(scores[np.arange(N),y]/sum_scores)
    loss=-np.sum(scores)
    loss=loss/N+reg*np.sum(W*W)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW

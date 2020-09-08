#Image classification notes

##nearest neighbor classification
```python
nn = NearestNeighbor() # create a Nearest Neighbor classifier class
nn.train(Xtr_rows, Ytr) # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows)# predict labels on the test images
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print 'accuracy: %f' % ( np.mean(Yte_predict == Yte) )
```
a simple Nearest Neighbor classifier using L1 distance:
```python
import numpy as np

class NearestNeighbor(object):
  def __init__(self):
    pass

  def train(self, X, y):
    """ X is N x D where each row is an example. Y is 1-dimension of size N """
    # the nearest neighbor classifier simply remembers all the training data
    self.Xtr = X
    self.ytr = y

  def predict(self, X):
    """ X is N x D where each row is an example we wish to predict label for """
    num_test = X.shape[0]
    # lets make sure that the output type matches the input type
    Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

    # loop over all test rows
    for i in range(num_test):
      # find the nearest training image to the i'th test image
      # using the L1 distance (sum of absolute value differences)
      distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
      min_index = np.argmin(distances) # get the index with smallest distance
      Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

    return Ypred

```

##Ways of computing distances between vectors
```python
#L1 distance
distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
```
```python
#L2 distance
distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
```
##K-Nearest Neighbor Classifier
Find k nearest neighbors in training set and vote for the label.
When k = 1, we recover the NN.

## Validation sets for Hyperparameter tuning
Split training set in two: a slightly smaller training set, and what we call a validation set.
在训练集中取得比较小的一部分数据作为验证集,剩下的训练数据进行模型的训练,用验证集来对模型进行验证,在结果中选择误差最小的模型(参数)

```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```

##Cross-validation
A more sophisticated technique for hyperparameter tuning
训练集比较小的时候适用
对训练集均分成N份,留一份做验证,剩下的做训练,这样每个模型(参数)对应N个不同误差,取平均.最后选择误差最小的模型(参数).

##NN分类器总结
基于原始像素的比较(L1/L2),对于高维的图像数据不适用;
计算复杂度都集中在test阶段

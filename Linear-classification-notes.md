#Linear classification notes

##Score function
mapping from images to label scores
f(xi,W,b)=Wxi+b

##Loss function
### SVM loss

##Regularization loss
Loss = data loss(e.g. SVM) + regularization loss(e.g. L2 norm)

## Pratical considerations
### Setting Delta
Delta的一般设定为1.0, delta和lambda看上去像是两个hyperparameters, 实际上两者同时影响着一个tradeoff, 那就是data loss和regu loss. 因为W的量级对scores有着直接的影响, 我们把W缩小, scores也变小, 我们把W变大, scores也变大, 所以分数之间的margin某种程度上也就没有意义. 它实际上取决于我们允许W在什么样的量级之上(由lambda决定)



##SVM vs. Softmax
SVM: score function + hinge loss
Softmax: score function + softmax function + cross-entropy loss

![avatar](https://cs231n.github.io/assets/svmvssoftmax.png)


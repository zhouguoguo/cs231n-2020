#Optimization notes

W的每一个维度,都是一个bowl-shaped形状.

hinge lost的定义形式,决定了任何一个维度的W, 要么前面有个正号(对应到错误的类别), 要么前面有个负号(对应到正确的类别).

所有的L加起来之后,对于任意维度上的W来说, 每一项要么是与该W无关,要么是关于该W的线性函数(位于0之上的部分)

## Strategy1:Random search
## Strategy2:Random Local Search
## Strategy3:Following the gradient
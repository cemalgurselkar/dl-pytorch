import torch

#ReLu Activations --> max(0,x)
def relu(x):
    return torch.max(torch.tensor(0,0),x)

#Leaky ReLu ----> max(ax, x) ---> variant of ReLu that allows a small, non-zero gradient
#when the input is negative, defined as y = max(ax,x), where a is a small positive constant.ss
def leaky_relu(x,alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

#Sigmoid ---> used for classification. output has value between 0 and 1.
def sigmoid(x):
    return 1/ (1 + torch.exp(-x))

#Softmax Activations ---> calculating each possibility of all statements.
def softmax(x, dim=None):
    if dim is None:
        dim = 0
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
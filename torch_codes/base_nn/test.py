from torch import nn
import numpy as np
import torch
from torch.autograd import Variable



def bce_loss_test():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2),
        nn.ReLU()
    )

    x = np.array([np.random.rand(3, 224, 224)])
    print(x)
    # 1, 3, 224, 224
    print(x.shape)

    m = nn.Sigmoid()
    # loss = nn.BCELoss(size_average=False, reduce=False)
    loss = nn.BCELoss()

    target = torch.Tensor(np.array([np.random.rand(3, 224, 224)]))

    sigmoid_out = m(torch.Tensor(x))
    out = loss(sigmoid_out, target)
    print('BCE loss: ', out)

    calculate_hand_loss = - (target * np.log(sigmoid_out) + (1 - target) * np.log(1 - sigmoid_out))
    print('Hand cal BCE loss: ', calculate_hand_loss)


def conv_test():
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.ReLU()
    )
    x = np.array([np.random.rand(3, 512, 512)])
    x = torch.Tensor(x)
    out = model(x)
    print(out)
    print(out.size())

def onehot_test():
    """
    test one-hot usage in torch
    :return:
    """
    depth = 5
    batch = torch.Tensor([1, 2, 4]).long()
    ones = torch.eye(depth)
    print(ones.index_select(0, batch))

def cross_entropy():
    loss = nn.CrossEntropyLoss()
    input = torch.randn(3, 5, requires_grad=True)
    print('input: ', input)
    target = torch.empty(3, dtype=torch.long).random_(5)
    print('target: ', target)
    output = loss(input, target)
    print('output: ', output)


if __name__ == '__main__':
    cross_entropy()
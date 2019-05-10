from torch import nn
import numpy as np
import torch
from torch.autograd import Variable
import cv2
from PIL import Image
# np.set_printoptions(threshold=np.inf)

from dataset.seg_cityscapes import id2label, trainId2label, id2trainId

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

def onehot_test():q
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


def mask_test():
    def map_func(x):
        return id2trainId[x]
    a = '/media/jintain/sg/permanent/datasets/Cityscapes/gtFine/train/ulm/ulm_000094_000019_gtFine_labelIds.png'
    # b = cv2.imread(a, cv2.IMREAD_GRAYSCALE)
    # print(b)
    # c = b

    c = np.asarray(Image.open(a))
    print(c)
    #
    cv2.imshow('ori', c)
    # cv2.waitKey(0)

    print(np.max(c), np.min(c))

    print(c.shape)
    print(id2trainId)
    vf = np.vectorize(map_func)
    c = np.array(vf(c), dtype=np.uint8)
    # c = np.array(map(id2trainId, c))
    print(c)
    print(np.max(c), np.min(c))



    # c = cv2.resize(c, dsize=(800, 500))
    cv2.imshow('', c)
    cv2.waitKey(0)
    print(c.shape)


def im_read_test():
    """
    In png label data, better using Image to read
    I still don't know how to read in opencv
    TODO: Fucking read index with opencv!!!!!!!!!!!!!!!
    :return:
    """
    img_f = 'data/2007_000032.png'
    # a = Image.open(img_f)
    a = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)

    a = np.asarray(a, dtype=np.int8)

    print(a)
    print(a.shape)
    print('max: ', np.max(a))
    # b = Image.open(img_f)
    # b = np.array(b, dtype=np.int8)
    # print(b)
    # print(b.shape)
    # print('max: ', np.max(b))
    # filt = (b == 15)
    # b = b[filt]
    # b = np.asarray(b, dtype=np.int8)
    cv2.imshow('rr', a)
    cv2.waitKey(0)


def mul():
    a = np.array([
        [(2, 3, 4), (3, 4, 5)],
        [(2, 3, 4), (3, 4, 5)],
        [(2, 3, 4), (3, 4, 5)]])
    b = np.array([[0, 0],
         [1, 0],
         [0, 1]])
    print(a@b)
    print(a[np.where(b > 0)])
    print(a*b)

if __name__ == '__main__':
    mask_test()
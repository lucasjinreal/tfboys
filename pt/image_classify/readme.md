# base_nn

A tiny network practise package. I do neural practise everyday here, written down some findings, spark lights and bugs all about computer vision or many other AI applications.



## Classification

The first level in AI, is complete image classification program. However, it not as easy as you think. If you write it from scratch, you will found that is not easy. I just write a MobileNet V2 version and training an image classifier on flowers datasets. But some problem got.

1. loss just not decrease.

   this is very strange. possiable reason for this is: image should be normalized. But how to normalize exactly?

2. loss function problem.

   still have 
   
   
## Seg

*2018.12.5*

now I think I am fix the loss problem, the best way to write FCN loss is using **NLLLoss +
LogSoftmax** inside pytorch, NLLLoss2d has been deprecated. And there is no CrossEntropy2D method
for 2D crossentropy calculation. 

After all, seems the loss is not stable when training. I start using only 6 classes in cityscapes to
train a segmentation net. which is:

- 1 -> 0 road
- 2 -> 11 person
- 3 -> 12 rider
- 4 -> 13 car
- 5 -> 14 truck
- 0: others

I think there must be some class weight balancing problem, if the loss is not dropping.

do some segmentation experiment on cityscapes and voc, but got some un-ususally result.

1. pytorch loss function error

    it means that label can not contains 255 or -1
   
 

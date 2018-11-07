# base_nn

A tiny network practise package. I do neural practise everyday here, written down some findings, spark lights and bugs all about computer vision or many other AI applications.



## Classification

The first level in AI, is complete image classification program. However, it not as easy as you think. If you write it from scratch, you will found that is not easy. I just write a MobileNet V2 version and training an image classifier on flowers datasets. But some problem got.

1. loss just not decrease.

   this is very strange. possiable reason for this is: image should be normalized. But how to normalize exactly?

2. loss function problem.

   still have 
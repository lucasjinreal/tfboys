## 0. Structure

Before we get started, let's take a deep look how shall we do this:

![](https://img-blog.csdn.net/20180417200420732?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2UwMTUyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


I take it into 4 parts, and that should enough. which are:

1. base network backend for feature extraction:

   this part we can using VGG16, MobileNet (Maybe?), ResNet, etc. so make it separateble and module.

2. RPN network:

   this part should return proposals, which is region of interests (rois). the output is actually already box with probabilities. rois will be used by regression subnet.

3. Regression Subnet:

   This part is actually how to classify rois, and regression the accurate location of boxes.

4. Dataloader:

   this part should be easy to done.



## 1. Dataloader

So, we should start from data, how to write a properly dataloader for your detection network feed? dataset is easy to writing, but how to prepare the labels? the labeled one is (x1, y1), (x2, y2) by default. How to 
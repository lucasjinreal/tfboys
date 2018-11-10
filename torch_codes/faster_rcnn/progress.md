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

So, we should start from data, how to write a properly dataloader for your detection network feed? dataset is easy to writing, but how to prepare the labels? the labeled one is (x1, y1), (x2, y2) by default. How to resize image and according process to box label?

Strategies are:

1. `max_side <= 1000` and `min_size <= 600`, got the scale and do same resize to box;
2. PIL read image should not need to normalize it again.

What we should provide to network inputs:

1. `image`: [batch_size, 788, 568, 3];
2. `bboxes`: [batch_size, 4], every example is `(y_min, x_min, y_max, x_max)` (should be this format???)



## 2. Feature Extractor

the base net before RPN should be module. We can using VGG, ResNet at least, or simply using a Mobilenet to integrate with it.

more detail to be add....



## 3. RPN

anchor setting. how should we set anchors? maybe using default settings?

- anchors:

  size are: 128, 256, 512, ratio are: 1:1, 1:2, 2:1 (maybe we can add more ratio). then there is a big step is using those anchors to **sample** on feature map. and get the **anchored_features**.

  ```python
  def get_anchored_features(anchors):
  	.....
  ```

- RPN training process:

  for train RPN, what loss should we gather. Other words, what's the label and what's the output what loss function should we use?

  After featuremap extractor got features, say 32x32x512, we got 512 channels, we using 1x1 convolution make the output to be 32x32x18 and 32x32x36.

  Why is there should be **18** and **36**, we have 9 anchors (3 size and 3 ratio), for every anchor,  we have 2 classes to predict, object or background. and 4 coordinates.

  So in RPN process, the gt_label is 0 or 1, indicates the anchor area is background or object. gt_loc is the area coordinates. We limited positive samples and negative samples to 3: 1. Positive samples no more than 128.

  Using cross-entropy-loss calcilate the category loss, and using smotth-l1-loss to calculate the location loss. Only do loc loss with positive objects not for the background.

- RPN generates ROIs

  Why should we need ROIs? **Not only need, but even more important than you think**. Without ROIs, we can not regress on it, and classify the features. However, if you finish the RPN part, the rest is very simple now. 

  We can call the post-process **ROI-Head**. This part purpose is regression the coordicates again and get the exactly category of that objects.


##  4.ROI-Head (Regression Subnet)

Now that we are in the last part. The rest of whole thing becomes simple.What we need, is just the output of RPN. and construct some layers with a simple **ROI Pooling** and some fully convolution layers.

The structure just like this:

![](https://img-blog.csdn.net/201804172012459?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2UwMTUyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

The ROI Pooling takes into feature map, but **we need map ROI locations back to feature map** and send that part into ROI-Head.

Now, let's talk about ROI Pooling layer, **what does it using for?** We have multiple size ROIs, but in order to send them into a FC layers, we have to fix it into certain size. 

![](https://img-blog.csdn.net/20180417201336327?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2UwMTUyOA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Like figure above, we have 2 ROIs which have the different size, but after ROI Pooling, we will have same size features. In a words, we simply **grid any size ROIs, and take the max pooling of every grid cell, and then get the same size output**.



## Training


记录TensorFlow中的两个坑

1. `tf.image.decode_image` 不能得到图片的维度信息，导致后面resize的操作无法进行/。这个十分但疼。

2. `tf.image.resize_biili..` 反正就是必须知道维度才能resize。
3. 一旦所有的数据都是tensor，你就不能使用其他的东西，这就是为什么tensorflow要提供一个操作语言一样的东西。。。
4. 所有的数据都最好专程`tf.float32`的精度。不然网络处理一下卷及什么就崩盘了。
5. 那个图片预处理的函数很重要啊，一定要读取，然后归一化。
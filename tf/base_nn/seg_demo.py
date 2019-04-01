"""

Segmentation demo using FCN to seg road


"""
import numpy as np
import os
from util.image_inference import ImageInferEngine
from seg_kitti_road import load_vgg, layers
import tensorflow as tf
import cv2


class Demo(ImageInferEngine):

    def __init__(self, f):
        super(Demo, self).__init__(f, is_show=True, record=True)

        self.model_path = 'checkpoints/model_fcn_road.ckpt'
        self.vgg_path = os.path.join('./checkpoints', 'vgg')
        self.num_classes = 2
        self.image_shape = (160, 576)
        self._init_model()

    def _init_model(self):
        self.sess = tf.Session()
        # Predict the logits
        self.input_image, self.keep_prob, \
            vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(self.sess, self.vgg_path)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, self.num_classes)
        self.logits = tf.reshape(nn_last_layer, (-1, self.num_classes))

        saver = tf.train.Saver()
        saver.restore(self.sess, self.model_path)
        print("Restored the saved Model in file: %s" % self.model_path)

    def solve_a_image(self, img):
        # BGR default, but ned reshape
        img = cv2.resize(img, (self.image_shape[1], self.image_shape[0]))
        im_softmax = self.sess.run(
            [tf.nn.softmax(self.logits)],
            {self.keep_prob: 1.0, self.input_image: [img]})

        im_softmax = im_softmax[0][:, 1].reshape(self.image_shape[0], self.image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(self.image_shape[0], self.image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0]]))
        return mask

    def vis_result(self, img, net_out):
        # net_out is the mask
        mask = np.array(net_out, dtype=np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        img = cv2.addWeighted(img, 0.5, mask, 0.5, 1)
        return img


if __name__ == '__main__':
    f = '/media/jintain/sg/permanent/datasets/KITTI/videos/2011_09_26/2011_09_26_drive_0106_sync/image_02/combined_data.mp4'
    demo = Demo(f=f)
    demo.run()



"""
implementation of MobileNetV3 in TensorFlow 2.0 API

codes largely brorrowed from https://github.com/Bisonai/mobilenetv3-tensorflow
"""
import tensorflow as tf


# ------------------------------- Layers part -------------------------------


class Identity(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Identity")

    def call(self, input):
        return input


class ReLU6(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="Relu6")
        self.relu6 = tf.keras.layers.ReLU(max_value=6, name="ReLU6")

    def call(self, input):
        return self.relu6(input)


class HardSigmoid(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSigmoid")
        self.relu6 = ReLU6()

    def call(self, input):
        return self.relu6(input + 3.0) / 6.0


class HardSwish(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name="HardSwish")
        self.hard_sigmoid = HardSigmoid()

    def call(self, input):
        return input * self.hard_sigmoid(input)


class GlobalAveragePooling2D(tf.keras.layers.Layer):
    """Return output shape (batch_size, rows, cols, channels).
   `tf.keras.layer.GlobalAveragePooling2D` is (batch_size, channels),
    """

    def __init__(self):
        super().__init__(name="GlobalAveragePooling2D")

    def build(self, input_shape):
        pool_size = tuple(map(int, input_shape[1:3]))
        self.gap = tf.keras.layers.AveragePooling2D(
            pool_size=pool_size,
            name=f"AvgPool{pool_size[0]}x{pool_size[1]}",
        )

    def call(self, input):
        return self.gap(input)


class BatchNormalization(tf.keras.layers.Layer):
    """All our convolutional layers use batch-normalization
    layers with average decay of 0.99.
    """

    def __init__(self):
        super().__init__(name="BatchNormalization")
        self.bn = tf.keras.layers.BatchNormalization(
            momentum=0.99,
            name="BatchNorm",
        )

    def call(self, input):
        return self.bn(input)


class ConvBnAct(tf.keras.layers.Layer):
    def __init__(
            self,
            filters: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 0,
            act_layer: tf.keras.layers.Layer = tf.keras.layers.ReLU,
    ):
        super().__init__(name="ConvBnAct")

        if padding <= 0:
            self.pad = Identity()
        else:
            self.pad = tf.keras.layers.ZeroPadding2D(
                padding=padding,
                name=f"Padding{padding}x{padding}",
            )
        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=stride,
            name=f"Conv{kernel_size}x{kernel_size}",
        )

        self.norm = BatchNormalization()
        self.act = act_layer()

    def call(self, input):
        x = self.pad(input)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class Bneck(tf.keras.layers.Layer):
    def __init__(
            self,
            out_channels: int,
            exp_channels: int,
            kernel_size: int,
            stride: int,
            use_se: bool,
            nl: str,
            act_layer: tf.keras.layers.Layer = ReLU6,
    ):
        super().__init__(name="Bneck")

        self.out_channels = out_channels
        self.exp_channels = exp_channels
        self.kernel_size = kernel_size
        self.stride = stride

        if use_se:
            self.SELayer = SEBottleneck()
        else:
            self.SELayer = Identity()

        if nl.lower() == "re":
            self.nl_layer = ReLU6()
        elif nl.lower() == "hs":
            self.nl_layer = HardSwish()
        else:
            raise NotImplementedError

    def build(self, input_shape):
        self.use_res = self.stride == 1 and int(
            input_shape[3]) == self.out_channels

        self.pw1 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.exp_channels,
                    kernel_size=1,
                    strides=1,
                    name="Conv1x1",
                ),
                BatchNormalization(),
                self.nl_layer,
            ],
            name="PointWise",
        )

        dw_padding = (self.kernel_size - 1) // 2
        self.dw = tf.keras.Sequential(
            [
                tf.keras.layers.ZeroPadding2D(
                    padding=dw_padding,
                    name=f"Padding_{dw_padding}x{dw_padding}",
                ),
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size=self.kernel_size,
                    strides=self.stride,
                    name=f"DWConv{self.kernel_size}x{self.kernel_size}",
                ),
                BatchNormalization(),
                self.SELayer,
                self.nl_layer,
            ],
            name="DepthWise",
        )

        self.pw2 = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=self.out_channels,
                    kernel_size=1,
                    strides=1,
                    name="Conv1x1",
                ),
                BatchNormalization(),
            ],
            name="PointWise",
        )

    def call(self, input):
        x = self.pw1(input)
        x = self.dw(x)
        x = self.pw2(x)

        if self.use_res:
            return input + x
        else:
            return x


class SEBottleneck(tf.keras.layers.Layer):
    def __init__(
            self,
            reduction: int = 4,
    ):
        super().__init__(name="SEBottleneck")
        self.reduction = reduction

    def build(self, input_shape):
        input_channels = int(input_shape[3])

        self.se = tf.keras.Sequential(
            [
                GlobalAveragePooling2D(),
                tf.keras.layers.Dense(input_channels // self.reduction),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Dense(input_channels),
                HardSigmoid(),
            ],
            name="SEBottleneck",
        )

    def call(self, input):
        return input * self.se(input)


# ------------------------ utils part --------------------------------
def _make_divisible(v, divisor, min_value=None):
    """https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class LayerNamespaceWrapper(tf.keras.layers.Layer):
    """`NameWrapper` defines auxiliary layer that wraps given `layer`
    with given `namespace`. This is useful for better visualization of network
    in TensorBoard.
    Default behavior of namespaces defined with nested `tf.keras.Sequential`
    layers is to keep only the most high-level `tf.keras.Sequential` namespace.
    """
    def __init__(
            self,
            layer: tf.keras.layers.Layer,
            namespace: str,
    ):
        super().__init__(name=namespace)
        self.wrapped_layer = tf.keras.Sequential(
            [
                layer,
            ],
            name=namespace,
        )

    def call(self, input):
        return self.wrapped_layer(input)


# ----------------------- Net building part --------------------------
# TODO: convert model to Functional or Sequential so that can be saved as h5 model

class MobileNetV3Small(tf.keras.Model):
    def __init__(
            self,
            classes: int=1001,
            width_multiplier: float=1.0,
            scope: str="MobileNetV3",
            divisible_by: int=8,
    ):
        super(MobileNetV3Small, self).__init__(name=scope)

        # First layer
        self.first_layer = ConvBnAct(16, kernel_size=3, stride=2, padding=1, act_layer=HardSwish)

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out  SE      nl     s
            [ 3,  16,   16,  True,   "RE",  2 ],
            [ 3,  72,   24,  False,  "RE",  2 ],
            [ 3,  88,   24,  False,  "RE",  1 ],
            [ 5,  96,   40,  True,   "HS",  2 ],
            [ 5,  240,  40,  True,   "HS",  1 ],
            [ 5,  240,  40,  True,   "HS",  1 ],
            [ 5,  120,  48,  True,   "HS",  1 ],
            [ 5,  144,  48,  True,   "HS",  1 ],
            [ 5,  288,  96,  True,   "HS",  2 ],
            [ 5,  576,  96,  True,   "HS",  1 ],
            [ 5,  576,  96,  True,   "HS",  1 ],
        ]

        self.bneck = tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

            self.bneck.add(
                LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        nl=NL,
                    ),
                    namespace=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(576 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = tf.keras.Sequential(
            [
                ConvBnAct(penultimate_channels, kernel_size=1, stride=1, act_layer=HardSwish),
                SEBottleneck(),
                GlobalAveragePooling2D(),
                HardSwish(),
                ConvBnAct(last_channels, kernel_size=1, act_layer=HardSwish),
                ConvBnAct(classes, kernel_size=1, act_layer=HardSwish),
            ],
            name="LastStage",
        )

    def call(self, input, training=False):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x


class MobileNetV3Large(tf.keras.Model):
    def __init__(
            self,
            classes: int=1001,
            width_multiplier: float=1.0,
            scope: str="MobileNetV3Large",
            divisible_by: int=8,
    ):
        super().__init__(name=scope)

        # First layer
        self.first_layer = ConvBnAct(16, kernel_size=3, stride=2, padding=1, act_layer=HardSwish)

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out   SE      NL     s
            [ 3,  16,   16,   False,  "RE",  1 ],
            [ 3,  64,   24,   False,  "RE",  2 ],
            [ 3,  72,   24,   False,  "RE",  1 ],
            [ 5,  72,   40,   True,   "RE",  2 ],
            [ 5,  120,  40,   True,   "RE",  1 ],
            [ 5,  120,  40,   True,   "RE",  1 ],
            [ 3,  240,  80,   False,  "HS",  2 ],
            [ 3,  200,  80,   False,  "HS",  1 ],
            [ 3,  184,  80,   False,  "HS",  1 ],
            [ 3,  184,  80,   False,  "HS",  1 ],
            [ 3,  480,  112,  True,   "HS",  1 ],
            [ 3,  672,  112,  True,   "HS",  1 ],
            [ 5,  672,  160,  True,   "HS",  1 ],
            [ 5,  672,  160,  True,   "HS",  2 ],
            [ 5,  960,  160,  True,   "HS",  1 ],
        ]

        self.bneck = tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(exp * width_multiplier, divisible_by)

            self.bneck.add(
                LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        nl=NL,
                    ),
                    namespace=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(960 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = tf.keras.Sequential(
            [
                ConvBnAct(penultimate_channels, kernel_size=1, act_layer=HardSwish),
                GlobalAveragePooling2D(),
                HardSwish(),
                tf.keras.layers.Conv2D(filters=last_channels, kernel_size=1, name="Conv1x1"),
                HardSwish(),
                tf.keras.layers.Conv2D(filters=classes, kernel_size=1, name="Conv1x1"),
            ],
            name="LastStage",
        )

    def call(self, input, training=False):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers
from tensorflow.keras import Sequential

(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()

class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt',
    'Sneaker', 'Bag', 'Ankle boot'
]

print(train_images.shape)
print(train_labels.shape)

train_images = train_images / 255.0
test_images = test_images / 255.0

model = Sequential([
    layers.Flatten(input_shape=[28, 28]),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

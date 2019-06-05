import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt


def load_ds():
    examples, meta = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True)
    print(examples)
    print(meta)

    train_ds, val_ds = examples['train'], examples['validation']
    print(train_ds)
    print(val_ds)

    for a in train_ds.take(2):
        print(a['pt'])
        print(a['en'])


def train():
    ds = load_ds()


if __name__ == "__main__":
    train()
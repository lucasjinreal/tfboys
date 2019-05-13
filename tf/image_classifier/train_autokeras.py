"""
3 lines of codes build a image classifier
using Autokeras

AutoML is a great and strong tool for ML
"""
from autokeras as ak


def train():
    clf = ak.ImageClassifier()
    clf.fit(x_train, y_train)
    results = clf.predict(x_test)


if __name__ == '__main__':
    train()
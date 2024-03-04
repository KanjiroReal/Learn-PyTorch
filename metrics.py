import torch as t


def accuracy(y_true, y_pred):
    correct = t.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


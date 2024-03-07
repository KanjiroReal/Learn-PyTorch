import torch as t
from time import perf_counter as get_time


def train_step(model: t.nn.Module, data_loader: t.utils.data.DataLoader, loss_func: t.nn.Module,
               optimizer: t.optim.Optimizer, device: t.device):
    """
    Performs training step with model trying to learn on data_loader
    :param model:
    :param data_loader:
    :param loss_func:
    :param optimizer:
    :param device:
    :return:
    """

    model.train()
    train_loss = 0
    # loop through batches
    for batch, (X, y) in enumerate(data_loader):
        start = get_time()
        # put data on the target device
        X, y = X.to(device), y.to(device)

        # forward
        y_pred = model(X)

        # loss (with metrics) per batch
        loss = loss_func(y_pred, y)
        train_loss += loss

        # zero grad
        optimizer.zero_grad()

        # backward
        loss.backward()

        # step
        optimizer.step()

        end = get_time()
        # print
        if batch % 10 == 0:
            print(
                f"Train step: {batch * len(X)}/{len(data_loader.dataset)} | {end - start:.3f}s/batch | Loss: {train_loss / 10:.5f}",
                end="\r")

    train_loss /= len(data_loader)

    # print(f"Train loss: {train_loss:.5f} |")
    return train_loss


def validation_step(model: t.nn.Module, data_loader: t.utils.data.DataLoader, loss_func: t.nn.Module, device: t.device):
    """

    :param device:
    :param model:
    :param data_loader:
    :param loss_func:
    :return:
    """

    model.eval()
    val_loss = 0
    with t.inference_mode():
        for batch, (X, y) in enumerate(data_loader):
            start = get_time()
            # send the data to the target device
            X, y = X.to(device), y.to(device)

            #  forward
            val_pred = model(X)

            # loss (and metrics) per batch
            val_loss += loss_func(val_pred, y)
            end = get_time()

            if batch % 10 == 0:
                print(
                    f"Test step: {batch * len(X)}/{len(data_loader.dataset)} | {end - start:.3f}s/batch | val_loss: {val_loss / 10:.5f}",
                    end="\r")

        val_loss /= len(data_loader)

        # print(f"Test loss: {test_loss:.5f} |\n")
        return val_loss

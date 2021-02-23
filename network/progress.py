import numpy as np


class Progress:
    def __init__(self, epochs):
        self.epochs = epochs
        self.epoch_train_losses = np.array([])
        self.epoch_val_loss = None


    def batch_train_loss(self, loss):
        self.epoch_train_losses = np.append(self.epoch_train_losses, loss)


    def val_loss(self, loss):
        self.epoch_val_loss = loss


    def print_epoch(self, epoch):
        epoch_avg_train_loss = np.mean(self.epoch_train_losses) if len(self.epoch_train_losses) > 0 else -1
        epoch_val_loss = self.epoch_val_loss if self.epoch_val_loss is not None else -1
        train_loss = "{:.5f}".format(epoch_avg_train_loss)
        val_loss = "{:.5f}".format(epoch_val_loss)
        print(
            f'Epoch {epoch}/{self.epochs}    |    Train loss: {train_loss}   -   Val loss: {val_loss}'
        )

        # clear values
        self.epoch_train_losses = np.array([])
        self.epoch_val_loss = None


    """
    def progress_bar(epochs, current_epoch, minibatches, current_minibatch):
        s = '['
        minibatches_chunk = ceil(minibatches / 6)
        freq = minibatches // minibatches_chunk
    
        if current_minibatch == 0 or freq % current_minibatch != 0:
            return
    
        bar_len = epochs * 6
        progress = (current_epoch * 6) + (current_minibatch // minibatches_chunk)
        s += '=' * progress
        s += ' ' * (bar_len - progress)
        s += ']'
        print(s, current_minibatch)
    """
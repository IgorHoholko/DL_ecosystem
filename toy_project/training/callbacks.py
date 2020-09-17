import wandb
import numpy as np
import datetime

def _get_current_time():
    return datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")

class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks=None, queue_length=10):
        callbacks = callbacks or []
        self.callbacks = [c for c in callbacks]
        self.queue_length = queue_length

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        # logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        # logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = _get_current_time()
        for callback in self.callbacks:
            callback.on_train_end(logs)


class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


# class WandbImageLogger(Callback):
#     """Custom callback for logging image predictions"""
#
#     def __init__(self, model, data_loader, example_count: int = 4):
#         super().__init__()
#         self.model_wrapper = model
#         self.val_images = [data_loader.dataset[i]['image']
#                            for i in np.random.randint(0, len(data_loader.dataset), example_count)]
#
#     def on_epoch_end(self, epoch, logs=None):
#         images = [
#             wandb.Image(image, caption="{}: {}".format(*self.model_wrapper.predict_on_image(image)))
#             for i, image in enumerate(self.val_images)
#         ]
#         wandb.log({"examples": images}, commit=False)


class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self,
                 monitor='val_loss',
                 min_delta=10e-6,
                 patience=5):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.stopped_epoch = 0
        super(EarlyStopping, self).__init__()

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            pass
        else:
            # if improved..
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.wait = 1
                logs['loss_improved'] = True
            else:
                logs['loss_improved'] = False
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    logs['stop_training'] = True
                self.wait += 1

    def on_train_end(self, logs):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' %
                (self.stopped_epoch))

import copy


class EarlyStoping:
    def __init__(self, patience=5, min_delta=0, restore_best_weights=True):
        """
        Early stopping to stop training when validation loss does not improve.
        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change to qualify as an improvement.
            restore_best_weights (bool): Whether to restore model weights from the epoch with the best value of the monitored quantity.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.best_model = None
        self.counter = 0
        self.status = ""

    def __call__(self, model, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
        elif self.best_loss - val_loss >= self.min_delta:
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_loss = val_loss
            self.counter = 0
            self.status = "Improvement found, counter reset"
        else:
            self.counter += 1
            self.status = "No improvement in the last {} epochs".format(self.counter)
            if self.counter >= self.patience:
                self.status = "Early stopping after {} epochs".format(self.patience)
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
        return False
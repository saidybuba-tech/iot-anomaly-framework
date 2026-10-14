from typing import Optional

import numpy as np

from .base_model import BaseModel


class AutoencoderModel(BaseModel):
    """Deep autoencoder trained to reconstruct normal traffic.

    Anomaly score = mean squared reconstruction error (formula (2) in the paper).
    """

    def __init__(self, hidden_dims=(32, 16), encoding_dim: int = 8, epochs: int = 50,
                 batch_size: int = 256, patience: int = 5, random_state: int = 42):
        self.hidden_dims = list(hidden_dims)
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state
        self.model = None

    def _build(self, input_dim: int):
        import tensorflow as tf
        from tensorflow.keras import Model, layers
        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(input_dim,))
        x = inp
        for h in self.hidden_dims:
            x = layers.Dense(h, activation="relu")(x)
        x = layers.Dense(self.encoding_dim, activation="relu")(x)
        for h in reversed(self.hidden_dims):
            x = layers.Dense(h, activation="relu")(x)
        out = layers.Dense(input_dim, activation="linear")(x)
        self.model = Model(inp, out)
        self.model.compile(optimizer="adam", loss="mse")

    def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, X_val: Optional[np.ndarray] = None) -> None:
        from tensorflow.keras.callbacks import EarlyStopping
        if y is not None:
            X = X[y == 0]  # reconstruct normal traffic only
        self._build(X.shape[1])
        kwargs = {}
        if X_val is not None:
            kwargs = dict(validation_data=(X_val, X_val),
                          callbacks=[EarlyStopping(monitor="val_loss", patience=self.patience,
                                                   restore_best_weights=True)])
        self.model.fit(X, X, epochs=self.epochs, batch_size=self.batch_size, verbose=0, **kwargs)

    def predict(self, X: np.ndarray) -> np.ndarray:
        rec = self.model.predict(X, verbose=0)
        return np.mean(np.square(X - rec), axis=1)

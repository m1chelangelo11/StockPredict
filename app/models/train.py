import sys
import os

sys.path.insert(0, os.path.abspath("."))
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
import numpy as np
import matplotlib.pyplot as plt
from app.config import MODEL_DIR, DATA_DIR


class trainModel:
    def __init__(self, lr=0.001, num_epochs=200, device=None):
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.criterion = nn.MSELoss()

        # miejsce na zapisywanie przebiegu uczenia
        self.train_losses = []
        self.val_losses = []
        self.val_predictions = []

        self.scaler_y = MinMaxScaler()

    def setModel(self, model):
        self.model = model
        self.setOptimizer()

    def setOptimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def load_data(self, ticker):
        # Implementacja ładowania danych z pliku
        torch.set_default_dtype(torch.float64)
        path = os.path.join(DATA_DIR, f"{ticker}_processed")
        y_train = torch.from_numpy(
            np.load(os.path.join(path, f"y_train_{ticker}.npy"))
        ).to(self.device)
        y_val = torch.from_numpy(np.load(os.path.join(path, f"y_val_{ticker}.npy"))).to(
            self.device
        )
        y_test = torch.from_numpy(
            np.load(os.path.join(path, f"y_test_{ticker}.npy"))
        ).to(self.device)

        # Konwertuj tensory na numpy dla skalera (scaler potrzebuje danych na CPU)
        y_train_cpu = y_train.cpu().numpy()
        y_val_cpu = y_val.cpu().numpy()
        y_test_cpu = y_test.cpu().numpy()

        self.scaler_y.fit(y_train_cpu)
        y_train_scaled = self.scaler_y.transform(y_train_cpu)
        y_val_scaled = self.scaler_y.transform(y_val_cpu)
        y_test_scaled = self.scaler_y.transform(y_test_cpu)

        # Konwertuj z powrotem na tensory i przenieś na device
        y_train = torch.from_numpy(y_train_scaled).to(self.device)
        y_val = torch.from_numpy(y_val_scaled).to(self.device)
        y_test = torch.from_numpy(y_test_scaled).to(self.device)

        X_train = torch.from_numpy(
            np.load(os.path.join(path, f"X_train_{ticker}.npy"))
        ).to(self.device)
        X_val = torch.from_numpy(np.load(os.path.join(path, f"X_val_{ticker}.npy"))).to(
            self.device
        )
        X_test = torch.from_numpy(
            np.load(os.path.join(path, f"X_test_{ticker}.npy"))
        ).to(self.device)

        return y_train, y_val, y_test, X_train, X_val, X_test

    def train(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

        for epoch in range(self.num_epochs):
            self.model.train()
            y_pred = self.model(X_train)
            loss = self.criterion(y_pred, y_train)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_losses.append(loss.item())

            log = f"Epoch {epoch}: Train Loss = {loss.item():.4f}"

            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val).item()
                    self.val_losses.append(val_loss)
                    # Przenieś predykcje na CPU przed konwersją do numpy
                    self.val_predictions.append(val_pred.cpu().detach().numpy())

                    log += f", Val Loss = {val_loss:.4f}"

            if epoch % 25 == 0:
                print(log)

    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            y_pred_scaled = self.model(X).cpu().numpy()
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred

    def evaluate(self, y_pred, y_true_scaled):
        y_true = self.scaler_y.inverse_transform(y_true_scaled.cpu().numpy())
        # MAE – sredni blad bezwzględny
        # RMSE – srednia oczekiwana roznice +/- miedzy wartoscia przewidywana a rzeczywista
        # MAPE – sredni procentowy blad bezwzgledny
        rmse = root_mean_squared_error(
            y_true, y_pred
        )  # Usunięto niepotrzebny np.sqrt()
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return rmse, mae, mape

    def display(self, y_test, y_test_pred, test_rmse, test_mae, test_mape):
        # Konwertuj tensory na numpy dla matplotlib
        if isinstance(y_test, torch.Tensor):
            y_test = self.scaler_y.inverse_transform(y_test.cpu().numpy())

        fig = plt.figure(figsize=(10, 8))
        gs = fig.add_gridspec(4, 1)
        ax1 = fig.add_subplot(gs[:3, 0])
        ax1.plot(y_test, color="blue", label="Rzeczywista cena")
        ax1.plot(y_test_pred, color="green", label="Predykcja ceny")
        ax1.legend()
        plt.title(" Stock price prediction ")
        plt.xlabel("Data")
        plt.ylabel("Cena")
        ax2 = fig.add_subplot(gs[3, 0])
        ax2.axhline(test_rmse, color="blue", linestyle="--", label="RMSE")
        ax2.axhline(test_mae, color="green", linestyle="--", label="MAE")
        ax2.axhline(test_mape, color="purple", linestyle="--", label="MAPE")
        ax2.plot(abs(y_test - y_test_pred), "r", label=" blad predykcji")
        ax2.legend()
        plt.title("blad predykcji")
        plt.xlabel("Data")
        plt.ylabel("Error")
        plt.tight_layout()
        plt.show()

    def save_model(self, path="model.pt"):
        filepath = os.path.join(MODEL_DIR, path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, model_class, path="model.pt", *args, y_train=None, **kwargs):
        filepath = os.path.join(MODEL_DIR, path)
        self.model = model_class(*args, **kwargs)
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        self.setOptimizer()
        if y_train is not None:
            if isinstance(y_train, torch.Tensor):
                y_train = y_train.cpu().numpy()
            self.scaler_y.fit(y_train)

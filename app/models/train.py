import sys
import os
sys.path.insert(0, os.path.abspath('.'))
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error,mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from app.config import MODEL_DIR

class trainModel:
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    def __init__(self, lr=0.001, num_epochs=200, device=None):
        self.lr = lr
        self.num_epochs = num_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.criterion = nn.MSELoss()

        # miejsce na zapisywanie przebiegu uczenia
        self.train_losses = []
        self.val_losses = []
        self.val_predictions = []

    def setModel(self,model):
        self.model = model
        self.setOptimizer()

    def setOptimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def modifyData(self,y_train,y_val,y_test,X_train,X_val,X_test):
        # Dopasuj i przeksztalc y
        y_train = self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        y_val = self.scaler_y.transform(y_val.reshape(-1, 1))
        y_test = self.scaler_y.transform(y_test.reshape(-1, 1))

        # konwersja danych do float tensorow i przeniesienie na GPU (jesli dostępne)
        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(self.device)
        y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(self.device)
        X_train, y_train = X_train.to(self.device), y_train.to(self.device)

        return y_train,y_val,y_test,X_train,X_val,X_test

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
                    self.val_predictions.append(val_pred.cpu().detach().numpy())

                    log += f", Val Loss = {val_loss:.4f}"

            if epoch % 25 == 0:
                print(log)

    def predict(self, X):
        self.model.eval()
        X = X.to(self.device)
        with torch.no_grad():
            return self.model(X).cpu().detach().numpy()
        
    def evaluate(self, y_pred,y_test):
        y_pred_rescaled = self.scaler_y.inverse_transform(y_pred)
        y_true_rescaled = self.scaler_y.inverse_transform(y_test)
        # MAE – sredni blad bezwzględny
        # RMSE – srednia oczekiwana roznice +/- miedzy wartoscia przewidywana a rzeczywista
        # MAPE – sredni procentowy blad bezwzgledny
        rmse = np.sqrt(root_mean_squared_error(y_true_rescaled, y_pred_rescaled))
        mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
        mape = np.mean(np.abs((y_true_rescaled - y_pred_rescaled) / y_true_rescaled)) * 100

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAPE: {mape:.2f}%")

        return y_pred_rescaled, y_true_rescaled,rmse,mae,mape
    
    def display(self,y_test,y_test_pred,test_rmse,test_mae,test_mape):

        fig = plt.figure(figsize=(10,8))
        gs = fig.add_gridspec(4,1)
        ax1 = fig.add_subplot(gs[:3, 0])
        ax1.plot(y_test, color='blue', label='Rzeczywista cena')
        ax1.plot(y_test_pred, color='green', label='Predykcja ceny')
        ax1.legend()
        plt.title(" Stock price prediction ")
        plt.xlabel('Data')
        plt.ylabel('Cena')
        ax2 = fig.add_subplot(gs[3, 0])
        ax2.axhline(test_rmse, color = 'blue', linestyle='--', label = 'RMSE')
        ax2.axhline(test_mae, color = 'green', linestyle='--', label = 'MAE')
        ax2.axhline(test_mape, color = 'purple', linestyle='--', label = 'MAPE')
        ax2.plot(abs(y_test - y_test_pred), 'r', label=" blad predykcji")
        ax2.legend()
        plt.title("blad predykcji")
        plt.xlabel('Data')
        plt.ylabel('Error')
        plt.tight_layout()
        plt.show()

    def save_model(self, path="model.pt"):
        filepath = os.path.join(MODEL_DIR, path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)

    def load_model(self,model_class, path="model.pt", *args, **kwargs):
        filepath = os.path.join(MODEL_DIR, path)
        self.model = model_class(*args, **kwargs)
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        self.setOptimizer()

import sys
import os

sys.path.insert(0, os.path.abspath("."))
from app.models.model import modelDoPredykcji
from app.models.train import trainModel
from app.data.data_pipeline import data_pipeline

# ustaw ticker
ticker = "AAPL"

# zaladuj dane przez pipeline
data_pipeline(ticker)

# zainicjalizuj model
# input_dim = rozmiar wprowadzonych danych (29 dni)
# hidden_dim = liczba neuronow
# num_layers = liczba warstw modelu (glebszy model)
# output_dim = liczba cech po ktorej model sie uczy
# learning rate(lr) = jak szybko model sie uczy
# num_epochs = ilosc epok przez ktore model sie uczy
input_dim = 29
hidden_dim = 128
num_layers = 2
output_dim = 1
model = modelDoPredykcji(input_dim, hidden_dim, num_layers, output_dim)
lr = 0.001
num_epochs = 200
# ustawiamy learning rate i ilosc epok
trainer = trainModel(lr, num_epochs)
trainer.setModel(model)

# wczytanie danych
y_train, y_val, y_test, X_train, X_val, X_test = trainer.load_data(ticker)

# trenowanie modelu na podstawie danych
trainer.train(X_train, y_train, X_val, y_val)


# zapis modelu
trainer.save_model("model.pt")

# test loading model
trainer2 = trainModel(lr, num_epochs)
trainer2.load_model(
    modelDoPredykcji,
    "model.pt",
    input_dim,
    hidden_dim,
    num_layers,
    output_dim,
    y_train=y_train,
)

# przewidywanie cen na podstawie danych testowych
y_pred = trainer2.predict(X_test)

# ewaluacja danych wraz z obliczeniem bledow
rmse, mae, mape = trainer2.evaluate(y_pred, y_test)

# wyswietlenie danych na wykresie
trainer2.display(y_test, y_pred, rmse, mae, mape)

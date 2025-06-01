import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from app.models.model import modelDoPredykcji
from app.models.train import trainModel
from app.data.data_pipeline import data_pipeline 
from app.models.model import modelDoPredykcji
from app.models.train import trainModel 

# ustaw ticker
ticker = "AAPL"

# zaladuj dane przez pipeline
X_train, X_val, X_test, y_train, y_val, y_test = data_pipeline(ticker)

# zainicjalizuj model
#input_dim = rozmiar wprowadzonych danych (29)
#hidden_dim = liczba neuronow
#num_layers = liczba warstw modelu (glebszy model)
#output_dim = liczba cech po ktorej model sie uczy
# learning rate(lr) = jak szybko model sie uczy
# num_epochs = ilosc epok przez ktore model sie uczy
input_dim=X_train.shape[2]
hidden_dim=64
num_layers=2
output_dim=1
model = modelDoPredykcji(input_dim, hidden_dim, num_layers, output_dim)
lr = 0.005
num_epochs = 400
# trenuj model
trainer = trainModel( lr, num_epochs)
trainer.setModel(model)
#modyfikowanie danych z pipeline do poprawnego formatu
y_train,y_val,y_test,X_train,X_val,X_test = trainer.modifyData(y_train,y_val,y_test,X_train,X_val,X_test)

#trenowanie modelu na podstawie danych
trainer.train(X_train, y_train, X_val, y_val)


#zapis modelu
trainer.save_model("model.pt")

#test loading model

trainer2 = trainModel(lr, num_epochs)
trainer2.load_model(modelDoPredykcji,"model.pt",input_dim, hidden_dim, num_layers, output_dim)

#przewidywanie cen na podstawie danych testowych
y_pred = trainer2.predict(X_test)


#ewaluacja danych wraz z obliczeniem bledow
y_pred_rescaled, y_true_rescaled,rmse,mae,mape = trainer2.evaluate(y_pred,y_test)

#wyswietlenie danych na wykresie
trainer2.display(y_true_rescaled,y_pred_rescaled,rmse,mae,mape)

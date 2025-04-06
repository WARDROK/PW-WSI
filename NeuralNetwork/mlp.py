import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

data = pd.read_csv('data.csv')
X = data.drop(columns=["quality"]).values  
y = data["quality"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=4321)


model = MLPClassifier(max_iter=1000, hidden_layer_sizes=(128,64,32), batch_size=64, learning_rate_init=0.001)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)


accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)

print(f"Accuracy on train set: {accuracy_train:.4f}")
print(f"Accuracy on test set: {accuracy_test:.4f}")
print(f"Mean absolute error: {mae:.2f}")

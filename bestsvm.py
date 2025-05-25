from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from dataloader import load_data

X_train, X_test, y_train, y_test = load_data()

C_val = 2.5
gamma_val = 0.5


svm_model = SVC(C=C_val, gamma=gamma_val, kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)


y_pred = svm_model.predict(X_test)

# accurasy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (C={C_val}, gamma={gamma_val}): {test_accuracy:.3f}")

# pinakas matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

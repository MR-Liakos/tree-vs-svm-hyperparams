from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloader import load_data

X_train, X_test, y_train, y_test = load_data()


best_depth = 5

# train
model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy with max_depth={best_depth}: {acc:.3f}")

# matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)



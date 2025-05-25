from dataloader import load_data
from visualize import plot_decision_boundary
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_data()

params = [
    (0.1, 0.01),
    (0.5, 0.05),
    (1.0, 0.1),
    (1.5, 0.2),
    (2.5, 0.5),
    (10.0, 1.0),
    (30.0, 5.0),
    (100.0, 10.0)

]

for C, gamma in params:
    svm = SVC(C=C, gamma=gamma, kernel='rbf', random_state=42)
    svm.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, svm.predict(X_train))
    test_acc  = accuracy_score(y_test,  svm.predict(X_test))
    print(f"SVM C={C} γ={gamma} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")
    plot_decision_boundary(svm, X_test, y_test, title=f"SVM C={C}, γ={gamma}")



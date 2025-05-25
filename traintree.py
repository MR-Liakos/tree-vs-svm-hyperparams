from dataloader import load_data
from visualize import plot_decision_boundary
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = load_data()

for depth in [1, 3, 5,6, 8, 10,15,None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, tree.predict(X_train))
    test_acc  = accuracy_score(y_test,  tree.predict(X_test))
    print(f"Tree max_depth={depth} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f}")
    plot_decision_boundary(tree, X_test, y_test, title=f"Tree depth={depth}")




from sklearn.model_selection import GridSearchCV    

def grid_optimisation(model, X_train, y_train, scoring="accuracy", cv=5):
    param_grid = {
        'n_estimators': [100, 300, 500, 1000],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'max_depth': [None, 10, 20, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'max_leaf_nodes': [None, 10, 20, 50, 100, 250, 500],
        'min_impurity_decrease': [0, 0.0001, 0.001, 0.01, 0.05],
        'bootstrap': [True, False]
    }

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=2
    )

    print("=========================================")
    print("   STARTING GRID SEARCH   ")
    print("=========================================\n")
    grid.fit(X_train, y_train)

    print("=========================================")
    print("   BEST PARAMETERS FOUND   ")
    print("=========================================\n")

    print("=========================================")
    print("   BEST SCORE   ")
    print("=========================================\n")
    print(grid.best_score_)

    return grid.best_estimator_

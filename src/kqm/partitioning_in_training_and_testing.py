from sklearn.model_selection import train_test_split

def partitioning_in_training_and_testing(X_train, y_train, val_size, val_random_state):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=val_random_state)
    return X_train, X_val, y_train, y_val



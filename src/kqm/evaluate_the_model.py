
def evaluate_the_model(model, X_test, y_test, X_train, y_train, kernel = None):
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_accuracy)
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Train loss:', train_loss)
    print('Train accuracy:', train_accuracy)
    if kernel is not None:
        print(f'Sigma: {kernel.sigma.numpy()}')
    return test_loss, test_accuracy, train_loss, train_accuracy

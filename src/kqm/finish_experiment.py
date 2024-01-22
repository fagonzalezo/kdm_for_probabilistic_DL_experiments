import wandb
def finish_experiment(test_loss, test_accuracy, train_loss, train_accuracy): 
    wandb.log({"Train loss": train_loss, "Train accuracy": train_accuracy})
    wandb.log({"Test loss": test_loss, "Test accuracy": test_accuracy})
    wandb.finish()

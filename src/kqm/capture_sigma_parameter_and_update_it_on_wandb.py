import wandb

def capture_sigma_parameter_and_update_it_on_wandb(sigma, kernel):
    sigma = kernel.sigma.numpy()
    print(f"Final sigma: {sigma}")

    wandb.config.update({"sigma" : sigma}, allow_val_change=True)
    return sigma

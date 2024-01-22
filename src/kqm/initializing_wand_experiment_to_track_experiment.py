import wandb

def initializing_wand_experiment_to_track_experiment(config, architecture, dataset, n_comp, encoded_size):
    run = wandb.init(
        # set the wandb project where this run will be logged
        project="mindlab",
        # track hyperparameters and run metadata
        config=config,
        name=f"{config['architecture']}_dataset_{config['dataset']}_comp_{config['n_comp']}_encoded_size_{config['encoded_size']}"
    )
    return run



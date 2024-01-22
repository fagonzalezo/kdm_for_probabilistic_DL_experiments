def capture_config(config):
    return config["input_shape"], config["base_depth"], config["encoded_size"], config["n_comp"], 0.1, config["num_classes"],\
                config["learning_rate"], config["epochs"], config["batch_size"], config["verbose"], config["wandb_enabled"],\
                config["pretrain_encoder"]



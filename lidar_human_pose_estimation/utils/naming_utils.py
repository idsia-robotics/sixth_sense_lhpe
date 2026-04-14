def create_model_name(training_config, default_model_path):
    """
    Generate a model name based on the training configuration.

    Args:
        training_config (dict): The training configuration dictionary.
        model_dir (str): The base directory for the model.

    Returns:
        str: The generated model name.
    """
    # Start with the base model directory name
    model_name = str(default_model_path.stem)

    # Check for dummy loss usage
    if training_config["train_configs"]["dummy_model"]:
        dummy_type = training_config["train_configs"]["dummy_type"]
        model_name = model_name + f"_dummy_{dummy_type}"
        return default_model_path.with_name(model_name)
    else:
        # Include history parameters if available
        model_name = (
            model_name
            + f'_h_{training_config["history_parameters"]["length"]}_{training_config["history_parameters"]["stride"]}'
        )

        # Check if skip connections are used
        if training_config["fcn_configs"]["use_skip_connection"]:
            model_name = model_name + "_res"

    # Check if skip connections are used
    model_name = model_name + "_"
    if training_config["train_configs"]["loss_function"]["presence"]:
        model_name = model_name + "p"
    if training_config["train_configs"]["loss_function"]["distance"]:
        model_name = model_name + "d"
    if training_config["train_configs"]["loss_function"]["orientation"]:
        if training_config["train_configs"]["loss_function"]["bidirection"]:
            model_name = model_name + "bi"
        else:
            model_name = model_name + "o"
    return default_model_path.with_name(model_name)

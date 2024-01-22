import itertools

def generate_grid_config_from_base_and_array_parameter(original_dict, product_dict):
    if not original_dict or not product_dict:
        return {}
    keys = product_dict.keys()
    values = product_dict.values()
    array_product = [{name: dato for name,dato in zip(keys, datos)} \
                     for datos in itertools.product(*values)]
    settings = [dict(original_dict, **current_dict) for current_dict in array_product]
    return settings



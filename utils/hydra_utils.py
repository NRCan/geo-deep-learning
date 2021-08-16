import os
import hydra
import shutil


def save_useful_info():
    shutil.copytree(
        os.path.join(hydra.utils.get_original_cwd(), 'src'),
        os.path.join(os.getcwd(), 'code/src')
    )
    shutil.copy2(
        os.path.join(hydra.utils.get_original_cwd(), 'hydra_run.py'),
        os.path.join(os.getcwd(), 'code')
    )


def get_hydra_key(key_to_lookup, config, default=None, msg=None, delete=False):
    """Returns a value given a dictionary key, or the default value if it cannot be found.
    :param key_to_lookup: key in dictionary (e.g. generated from .yaml)
    :param config: (dict) dictionary containing keys corresponding to parameters used in script
    :param default: default value assigned if no value found with provided key
    :param msg: message returned with AssertionError si length of key is smaller or equal to 1
    :param delete: (bool) if True, deletes parameter, e.g. for one-time use.
    :return:
    """
    if key_to_lookup not in config.keys() or config.get(key_to_lookup) is None:
        # raise AssertionError("Must provide an existing keys")
        return default
    else:
        return config.get(key_to_lookup)

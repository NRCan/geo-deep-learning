from ruamel_yaml import YAML

def ReadParameters(ParamFile):
    """Read and return parameters in .yaml file
    Args:
        ParamFile: Full file path
    Returns:
        YAML (Ruamel) CommentedMap dict-like object
    """
    yaml = YAML()
    with open(ParamFile) as yamlfile:
        params = yaml.load(yamlfile)
    return params

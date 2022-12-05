import os
import time
import hydra
import logging

from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict
from utils.utils import print_config, get_git_hash


@hydra.main(config_path="config", config_name="gdl_config_template")
def run_gdl(cfg: DictConfig) -> None:
    """
    Function general for Geo Deep-Learning using Hydra library to rules all the
    parameters and functions using during the task wanted to execute.

    Process
    -------
    1. Read and convert the `gdl_config.yaml` to a dictionary.
    2. Verify if the code and other information need to be saved.
    3. If the debug option is activate, the entire config yaml will be printed
       and save in a log. In addition of that, if the mode is `train`, a
       validation run will precede the training to assure the well functioning
       of the code.
    4. Verify is the chosen mode is available.
    5. Verify is the chosen task is available and run the code link to that
       task.

    -------
    :param cfg: (DictConfig) Parameters and functions in the main yaml config
                file.
    """
    cfg = OmegaConf.create(cfg)

    # debug config
    if cfg.debug:
        pass
        # cfg.training.num_sanity_val_steps = 1  # only work with pytorch lightning
        # logging.info(OmegaConf.to_yaml(cfg, resolve=True))

    # check if the mode is chosen
    if type(cfg.mode) is DictConfig:
        msg = "You need to choose between those modes: {}"
        logging.critical(msg.format(list(cfg.mode.keys())))
        raise ValueError()

    # save all overwritten parameters
    logging.info('\nOverwritten parameters in the config: \n' + cfg.general.config_override_dirname)

    # Start -----------------------------------
    msg = "Let's start {} for {} !!!".format(cfg.mode, cfg.general.task)
    logging.info(
        "\n" + "-" * len(msg) + "\n" + msg +
        "\n" + "-" * len(msg)
    )
    # -----------------------------------------

    # Start the timer
    start_time = time.time()
    # Read the task and execute it
    task = get_method(f"{cfg.mode}_{cfg.general.task}.main")
    task(cfg)

    # Add git hash from current commit to parameters.
    with open_dict(cfg):
        cfg.general.git_hash = get_git_hash()

    # Pretty print config using Rich library
    if cfg.get("print_config"):
        print_config(cfg, resolve=True)

    # End --------------------------------
    msg = "End of {} !!!".format(cfg.mode)
    logging.info(
        "\n" + "-" * len(msg) + "\n" + msg + "\n" +
        "Elapsed time: {:.2f}s".format(time.time() - start_time) +
        "\n" + "-" * len(msg) + "\n"
    )
    # ------------------------------------


if __name__ == '__main__':
    run_gdl()

"""Logging configuration."""

import logging.config
from pathlib import Path

import yaml

script_dir = Path(__file__).resolve().parent
CONFIG_DIR = script_dir / "log_config.yaml"
# USER_CACHE = Path.home().joinpath(".cache")
# LOG_DIR = Path.cwd().joinpath("logs")
# LOG_DIR.mkdir(parents=True, exist_ok=True)

# timestamp = datetime.now(tz=ZoneInfo("America/New_York")).strftime("%Y%m%d-%H_%M_%S")
# logfilename = f"{LOG_DIR}/{timestamp}.log"

with (CONFIG_DIR).open("r") as f:
    config = yaml.safe_load(f.read())
    # config["handlers"]["file"]["filename"] = logfilename
    logging.config.dictConfig(config)

logger = logging.getLogger(__name__)

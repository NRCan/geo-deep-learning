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

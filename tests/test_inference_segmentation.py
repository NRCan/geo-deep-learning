import os
from pathlib import Path
from omegaconf import DictConfig
from inference_segmentation import main as run_inference

def test_inference_segmentation():
    """Test inference segmentation"""
    working_folder = "tests/data/inference"
    model_path = "tests/data/inference/test_model.pt"
    raw_data_csv = "tests/data/inference/test.csv"
    data_dir = "tests/data/inference"
    bands_requested = [1, 2, 3]
    output_mask = Path("tests/data/inference/test_mask.tif")
    
    cfg = {"general": {"project_name": "inference_segmentation_test"},
           "dataset": {"bands": bands_requested, 
                       "raw_data_dir": data_dir,},
           "inference": {"model_path": model_path, 
                         "raw_data_csv": raw_data_csv, 
                         "root_dir":working_folder},
           "tiling": {"clahe_clip_limit": 0},}
    
    cfg = DictConfig(cfg)
    run_inference(cfg)
    assert output_mask.exists()
    os.remove(output_mask)
    
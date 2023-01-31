# Inference config

## Input imagery (str)

> Input imagery should be either a csv of stac item

### raw_data_csv (str)

Points to a csv containing paths to imagery for inference. If a ground truth is present in 2nd column, it will be 
ignored.

### input_stac_item (str)

A path or url to [stac item](../../dataset/README.md#use-case-5-single-band-raster-files-identified-as-assets-in-a-stac-item) 
directly. See stac item example for [Spacenet test data](https://datacube-stage.services.geo.ca/api/collections/spacenet-samples/items/SpaceNet_AOI_2_Las_Vegas-056155973080_01_P001-WV03), 
also contained in [test data](../../tests/data/spacenet.zip).

### state_dict_path (str)

Path to checkpoint containing trained weights for a given neural network architecture.

## chunk_size (int)

Size of chunk (in pixels) to read use for inference iterations over input imagery. The input patch will be square, 
therefore "chunk_size = 512" will generate 512 x 512 patches.

## max_pix_per_mb_gpu (int)

Defaults to 25. 

If chunk_size is omitted, this defines a "maximum number of pixels per MB of GPU Ram" that should be considered. 
E.g. if GPU has 1000 Mb of Ram and this parameter is set to 10, chunk_size will be set to sqrt(1000 * 10) = 100.

> Since this feature is based on a rule-of-thumb and assumes some prior empirical testing. WIP. 
  
### prep_data_only (bool)

If True, the inference script will exit after preparation of input data:
1. If checkpoint path is url, then the checkpoint will be downloade;
2. If imagery points to urls, it will be downloaded;
3. If input model expects imagery with [histogram equalization](../tiling/README.md#clahe-clip-limit-int), this 
enhancement is applied and equalized images save to disk.

## GPU parameters

### gpu (int)

Number of gpus to use at inference.
> Current implementation doesn't support gpu>1

### max_used_perc (int)

If RAM usage of detected GPU exceeds this percentage, it will be ignored.

### max_used_ram (int)

If GPU's usage exceeds this percentage, it will be ignored

## Post-processing

### ras2vec (bool)

If True, a polygonized version of the inference (.gpkg) will be created with rasterio tools


from torchgeo.datasets import GeoDataset

class BlueSkyGeo(GeoDataset):
    def __init__(
            self,
            local_or_stac_tiff: str) -> None:
        """
        """
        self.item_url = item_path
        self.bands = bands
        if len(self.bands) == 0:
            logging.warning(f"At least one band should be chosen if assets need to be reached")
        self.root = Path(root)
        self.transforms = transforms
        self.separate_files = singleband_files
        self.download = download
        self.pad = pad
        self.outpath = outpath
        self.cache = download

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        self.item_url = self.item_url if is_url(self.item_url) else to_absolute_path(self.item_url)
        # Read Stac item from url
        if self.separate_files:
            self.item = SingleBandItemEO(pystac.Item.from_file(str(self.item_url)))
        else:
            raise NotImplementedError(f"Currently only support single-band Stac Items")  # TODO

        # Create band inventory (all available bands)
        self.all_bands = [band for band in self.item.asset_by_common_name.keys()]

        # Filter only desired bands
        self.bands_dict = {k: v for k, v in self.item.asset_by_common_name.items() if k in self.bands}

        # Make sure desired bands are subset of inventory
        if not set(self.bands).issubset(set(self.all_bands)):
            raise ValueError(f"Selected bands ({self.bands}) should be a subset of available bands ({self.all_bands})")

        # Download assets if desired
        if self.download:
            for cname in self.bands:
                out_name = self.root / Path(self.bands_dict[cname]['href']).name
                download_url(self.bands_dict[cname]['href'], root=str(self.root), filename=str(out_name))
                self.bands_dict[cname]['href'] = out_name

        # Open first asset with rasterio (for metadata: colormap, crs, resolution, etc.)
        if self.bands:
            self.first_asset = self.bands_dict[self.bands[0]]['href']
            self.first_asset = self.first_asset if is_url(self.first_asset) else to_absolute_path(self.first_asset)

            self.src = rio.open(self.first_asset)

            # See if file has a color map
            try:
                self.cmap = self.src.colormap(1)
            except ValueError:
                pass

            if crs is None:
                crs = self.src.crs
            if res is None:
                res = self.src.res[0]

            # to implement reprojection, see:
            # https://github.com/microsoft/torchgeo/blob/3f7e525fbd01dddd25804e7a1b7634269ead1760/torchgeo/datasets/geo.py#L361
            minx, miny, maxx, maxy = self.src.bounds

            # Get temporal information from STAC item
            self.date = self.item.item.datetime
            mint = maxt = self.date.timestamp()

            # Add paths to Rtree index
            coords = (minx, maxx, miny, maxy, mint, maxt)

            self.index.insert(0, coords, self.first_asset)
            self._crs = cast(CRS, crs)
            self.res = cast(float, res)

    
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # TODO: turn off external logs (ex.: rasterio._env)
        # https://stackoverflow.com/questions/35325042/python-logging-disable-logging-from-imported-modules
        with rio.Env(CPL_CURL_VERBOSE=False):
            if self.separate_files:
                data_list: List[Tensor] = []
                for band in getattr(self, "bands", self.all_bands):
                    band_filepaths = []
                    filepath = self.bands_dict[band]['href']  # hardcoded: stac item reader needs asset_by_common_name()
                    filepath = filepath if is_url(filepath) else to_absolute_path(filepath)
                    band_filepaths.append(filepath)
                    data_list.append(self._merge_files(band_filepaths, query))
                data = torch.cat(data_list)  # type: ignore[attr-defined]
            else:
                # FIXME: implement multi-band Stac item: https://github.com/stac-extensions/eo/blob/main/examples/item.json
                data = self._merge_files(filepaths, query)
        data = data.float()

        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query, "files": filepaths}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
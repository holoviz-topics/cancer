# Original from @flying-sheep 
# https://gist.github.com/flying-sheep/3ff54234019cc7c84e84cbbe649209c5

from dataclasses import dataclass
from enum import Enum, auto
from typing import cast, overload
import holoviews as hv
import scanpy as sc
import numpy as np

class Raise(Enum):
    Sentry = auto()

@dataclass
class AnnDataProxy:
    adata: sc.AnnData
    _column_cache: dict = None

    def __post_init__(self):
        self._column_cache = {}

    @overload
    def get(self, k: str, /, default: None = None) -> np.ndarray | None: ...
    @overload
    def get(self, k: str, /, default: np.ndarray | Raise) -> np.ndarray: ...
    def get(self, k: str, /, default: np.ndarray | Raise | None = None) -> np.ndarray | None:
        if k in self._column_cache:
            return self._column_cache[k]

        try:
            if "." not in k:
                if k not in self.adata.var_names:
                    if default is Raise.Sentry:
                        raise KeyError(k)
                    return default
                result = self.adata[:, k].X.flatten()
            else:
                attr_name, k = k.split(".", 1)
                if attr_name == "obs":
                    if k in self.adata.obs:
                        result = self.adata.obs[k].values
                    else:
                        if default is Raise.Sentry:
                            raise KeyError(k)
                        return default
                elif attr_name == "var":
                    if k in self.adata.var:
                        result = self.adata.var[k].values
                    else:
                        if default is Raise.Sentry:
                            raise KeyError(k)
                        return default
                elif attr_name == "obsm":
                    if "." in k:
                        obsm_key, idx = k.split(".", 1)
                        if obsm_key in self.adata.obsm:
                            idx = int(idx)
                            result = self.adata.obsm[obsm_key][:, idx]
                        else:
                            if default is Raise.Sentry:
                                raise KeyError(k)
                            return default
                    else:
                        if default is Raise.Sentry:
                            raise KeyError(k)
                        return default
                elif attr_name == "varm":
                    if "." in k:
                        varm_key, idx = k.split(".", 1)
                        if varm_key in self.adata.varm:
                            idx = int(idx)
                            result = self.adata.varm[varm_key][:, idx]
                        else:
                            if default is Raise.Sentry:
                                raise KeyError(k)
                            return default
                    else:
                        if default is Raise.Sentry:
                            raise KeyError(k)
                        return default
                else:
                    if default is Raise.Sentry:
                        raise KeyError(k)
                    return default

            self._column_cache[k] = result
            return result
            
        except (AttributeError, KeyError, ValueError, IndexError):
            if default is Raise.Sentry:
                raise KeyError(k)
            return default

    def __contains__(self, k: str) -> bool:
        try:
            return self.get(k, None) is not None
        except KeyError:
            return False

    def __getitem__(self, k: str) -> object:
        return self.get(k, Raise.Sentry)
    
    def __len__(self) -> int:
        return len(self.adata)

    def columns(self, dims=None):
        """Convert to dictionary of columns"""
        if dims is None:
            dims = []
        return {d: self.get(d) for d in dims if self.get(d) is not None}

class AnnDataInterface(hv.core.Interface):
    types = (sc.AnnData,)
    datatype = "anndata"

    @classmethod
    def init(cls, eltype, data: sc.AnnData | AnnDataProxy | list, kdims: list[str] | None, vdims: list[str] | None) -> tuple[AnnDataProxy, dict]:
        """Initialize data for the interface"""
        if isinstance(data, list):
            # If we get a list, assume it's aggregated data and convert to columns
            dims = (kdims or []) + (vdims or [])
            if len(data) == 1:
                data = {d: v for d, v in zip(dims, data[0])}
            else:
                data = {d: [v[i] for v in data] for i, d in enumerate(dims)}
            
            # Create a temporary AnnData object
            obs_data = {k.split('.')[-1]: v for k, v in data.items() if k.startswith('obs.')}
            if obs_data:
                temp_adata = sc.AnnData(np.zeros((len(next(iter(obs_data.values()))), 1)))
                for k, v in obs_data.items():
                    temp_adata.obs[k] = v
                data = temp_adata
        
        proxy = AnnDataProxy(data) if isinstance(data, sc.AnnData) else data
        return proxy, {'kdims': kdims, 'vdims': vdims}, {}
    
    @classmethod
    def validate(cls, dataset, vdims=True):
        """Validate that the dataset contains the specified dimensions."""
        dims = 'all' if vdims else 'key'
        dimensions = dataset.dimensions(dims, label='name')
        proxy = cast(AnnDataProxy, dataset.data)
        
        not_found = []
        for dim in dimensions:
            if proxy.get(dim, None) is None:
                not_found.append(dim)
                
        if not_found:
            raise Exception("Supplied data does not contain specified "
                          f"dimensions, the following dimensions were not found: {not_found!r}", cls)
    
    @classmethod
    def values(cls, data: hv.Dataset, dim: hv.Dimension | str, expanded=True, flat=True, compute=True, keep_index=False) -> np.ndarray:
        dim = data.get_dimension(dim)
        proxy = cast(AnnDataProxy, data.data)
        return proxy[dim.name]
    
    @classmethod
    def dimension_type(cls, data: hv.Dataset, dim: hv.Dimension | str) -> np.dtype:
        dim = data.get_dimension(dim)
        proxy = cast(AnnDataProxy, data.data)
        values = proxy[dim.name]
        if np.isscalar(values):
            return np.array([values]).dtype
        return values.dtype

    @classmethod
    def aggregate(cls, dataset, dimensions, function, **kwargs):
        """Aggregate over the supplied key dimensions with the specified function."""
        proxy = cast(AnnDataProxy, dataset.data)
        aggregated = {}
        
        for vd in dataset.vdims:
            data = proxy[vd.name]
            if data is not None:
                if function in [np.std, np.var]:
                    fn = lambda x: function(x, ddof=0)
                else:
                    fn = function
                aggregated[vd.name] = fn(data, **kwargs)
        
        return aggregated, []

    @classmethod
    def unpack_scalar(cls, dataset, data):
        """Unpack scalar values from the aggregated data."""
        if len(dataset.vdims) == 1:
            vdim = dataset.vdims[0].name
            if vdim in data and np.isscalar(data[vdim]):
                return data[vdim]
        return data

if AnnDataInterface.datatype not in hv.core.data.datatypes:
    hv.core.data.datatypes.append(AnnDataInterface.datatype)
hv.core.Interface.register(AnnDataInterface)
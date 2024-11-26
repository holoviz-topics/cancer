# import holoviews as hv
# import hvplot.pandas  # noqa
# import pandas as pd
# import param

# hv.extension("bokeh")


# class UMAPPlot(param.Parameterized):
#     """
#     A class for creating UMAP plots from AnnData objects.
#     """

#     adata = param.Parameter()
#     color = param.String(default="cell_type")
#     tools = param.ListSelector(default=["box_select", "lasso_select"])
#     width = param.Integer(default=500)
#     height = param.Integer(default=500)

#     def __init__(self, **params):
#         super().__init__(**params)
#         self._prepare_data()

#     def _prepare_data(self):
#         # Prepare UMAP data from adata object
#         self.obs_df = self.adata.obs.copy()
#         umap_df = pd.DataFrame(
#             self.adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=self.adata.obs_names
#         )
#         self.obs_df = self.obs_df.join(umap_df)

#     def plot(self):
#         # Generate UMAP scatter plot
#         return self.obs_df.hvplot.points(
#             x="UMAP1",
#             y="UMAP2",
#             color=self.color,
#             cmap="Category20",
#             responsive=True,
#             width=self.width,
#             height=self.height,
#             tools=self.tools,
#         ).opts(
#             xlabel="UMAP1",
#             ylabel="UMAP2",
#             framewise=True,
#             toolbar="above",
#             fontsize={"labels": 10, "ticks": 8},
#         )

import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element import Element2D
from holoviews.element.selection import Selection2DExpr
from holoviews.plotting.bokeh import PointPlot
import pandas as pd
import param


hv.extension('bokeh')

class UMAPPlot(Dataset, Element2D, Selection2DExpr):
    """
    Prepare UMAP plot data from AnnData objects.
    """

    group = param.String(default="UMAPPlot", constant=True)

    kdims = param.List(
        default=["UMAP1", "UMAP2"],
        bounds=(2, 2),
        doc="Key dimensions representing UMAP coordinates.",
    )

    vdims = param.List(
        default=[],
        doc="Value dimensions representing additional data to be plotted.",
    )

    color = param.String(default="cell_type", doc="Column to use for coloring points.")

    def __init__(self, adata, **params):
        self.adata = adata
        params = dict(params)
        self.color = params.pop("color", self.color)
        self._prepare_data()
        super().__init__(self.data, **params)

    def _prepare_data(self):
        # Prepare UMAP data
        if "X_umap" not in self.adata.obsm:
            raise ValueError("UMAP coordinates ('X_umap') not found in adata.obsm.")
        umap_df = pd.DataFrame(
            self.adata.obsm["X_umap"],
            columns=["UMAP1", "UMAP2"],
            index=self.adata.obs_names,
        )
        obs_df = self.adata.obs.copy()

        # Ensure the color column exists
        if self.color not in obs_df.columns:
            raise ValueError(f"Color column '{self.color}' not found in adata.obs.")
        self.vdims = [self.color] if self.color not in self.vdims else self.vdims

        self.data = obs_df.join(umap_df)

class UMAPPlotPlot(PointPlot):
    pass

hv.Store.register({UMAPPlot: UMAPPlotPlot}, backend="bokeh")

# Set default options
options = hv.Store.options(backend="bokeh")
options.UMAPPlot = hv.Options(
    "style",
    color="cell_type",
    cmap="Category20",
)

options.UMAPPlot = hv.Options(
    "plot",
    responsive=True,
    min_height=300,
    tools=["hover", "box_select", "lasso_select"],
    fontscale=0.7,
    xlabel="UMAP1",
    ylabel="UMAP2",
    show_legend=False,
)
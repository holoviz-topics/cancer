# import holoviews as hv
# from holoviews.core.data import Dataset
# from holoviews.element import Element2D
# from holoviews.element.selection import Selection2DExpr
# from holoviews.plotting.bokeh import PointPlot
# import pandas as pd
# import param

# hv.extension("bokeh")

# class DotPlot(Dataset, Element2D, Selection2DExpr):
#     """
#     Prepare dotplot data from AnnData objects
#     """

#     group = param.String(default="DotPlot", constant=True)

#     kdims = param.List(
#         default=["gene_id", "cluster"],
#         bounds=(2, 2),
#         doc="Key dimensions representing genes and clusters.",
#     )

#     vdims = param.List(
#         default=["mean_expression", "size", "percentage"],
#         doc="Value dimensions representing expression metrics.",
#     )

#     marker_genes = param.Dict(default=None, doc="Dictionary of marker genes.")
#     groupby = param.String(default="cell_type", doc="Column to group by.")
#     max_dot_size = param.Integer(default=10, doc="Maximum size of the dots.")
#     expression_cutoff = param.Number(default=0.1, doc="Cutoff for expression.")

#     def __init__(self, adata, **params):
#         self.adata = adata
#         params = dict(params)
#         self.marker_genes = params.get("marker_genes", self.marker_genes)
#         if self.marker_genes is None:
#             raise ValueError("marker_genes must be provided.")
#         self.groupby = params.get("groupby", self.groupby)
#         self.max_dot_size = params.get("max_dot_size", self.max_dot_size)
#         self.expression_cutoff = params.get("expression_cutoff", self.expression_cutoff)
#         self._prepare_data()
#         super().__init__(self.data, **params)

#     def _prepare_data(self):
#         self.all_marker_genes = list(
#             set(gene for genes in self.marker_genes.values() for gene in genes)
#         )
#         expression_data = self.adata[:, self.all_marker_genes].X
#         if hasattr(expression_data, "toarray"):
#             expression_data = expression_data.toarray()
#         expression_df = pd.DataFrame(
#             expression_data,
#             columns=self.all_marker_genes,
#             index=self.adata.obs_names,
#         )
#         obs_df = self.adata.obs.copy()
#         self.cells_df = obs_df.join(expression_df)

#         # Prepare data for the dot plot
#         dp_results = []
#         groupby = self.cells_df[self.groupby]
#         for gene in self.all_marker_genes:
#             for cluster, indices in groupby.groupby(
#                 groupby, observed=True
#             ).groups.items():
#                 cluster_data = self.cells_df.loc[indices, gene]
#                 percentage = (cluster_data > self.expression_cutoff).mean() * 100
#                 mean_expression = cluster_data.mean()
#                 dp_results.append(
#                     {
#                         "gene_id": gene,
#                         "cluster": cluster,
#                         "percentage": percentage,
#                         "mean_expression": mean_expression,
#                         "size": (percentage / 100) * self.max_dot_size,
#                     }
#                 )
#         dp_df = pd.DataFrame(dp_results)
#         dp_df["cluster"] = dp_df["cluster"].astype(str)
#         self.data = dp_df

import holoviews as hv
from holoviews.core.data import Dataset
from holoviews.element import Element2D
from holoviews.element.selection import Selection2DExpr
from holoviews.plotting.bokeh import PointPlot
import pandas as pd
import param

hv.extension("bokeh")


class DotPlot(Dataset, Element2D, Selection2DExpr):
    """
    Prepare dotplot data from AnnData objects.
    """

    group = param.String(default="DotPlot", constant=True)

    kdims = param.List(
        default=["gene_id", "cluster"],
        bounds=(2, 2),
        doc="Key dimensions representing genes and clusters.",
    )

    vdims = param.List(
        default=["mean_expression", "size", "percentage", "mean_expression_norm"],
        doc="Value dimensions representing expression metrics.",
    )

    marker_genes = param.Dict(default=None, doc="Dictionary of marker genes.")
    groupby = param.String(default="cell_type", doc="Column to group by.")
    max_dot_size = param.Integer(default=15, doc="Maximum size of the dots.")
    expression_cutoff = param.Number(default=0.1, doc="Cutoff for expression.")

    def __init__(self, adata, **params):
        self.adata = adata
        params = dict(params)
        self.marker_genes = params.pop("marker_genes", self.marker_genes)
        if self.marker_genes is None:
            raise ValueError("marker_genes must be provided.")
        self.groupby = params.get("groupby", self.groupby)
        self.max_dot_size = params.get("max_dot_size", self.max_dot_size)
        self.expression_cutoff = params.get("expression_cutoff", self.expression_cutoff)
        self._prepare_data()
        super().__init__(self.data, **params)

    def _prepare_data(self):
        # Flatten the list of marker genes
        self.all_marker_genes = list(
            set(gene for genes in self.marker_genes.values() for gene in genes)
        )

        # Check if all genes are present in adata.var_names
        missing_genes = set(self.all_marker_genes) - set(self.adata.var_names)
        if missing_genes:
            print(f"Warning: The following genes are not present in the dataset and will be skipped: {missing_genes}")
            # Remove missing genes from the list
            self.all_marker_genes = [gene for gene in self.all_marker_genes if gene in self.adata.var_names]
            if not self.all_marker_genes:
                raise ValueError("None of the specified marker genes are present in the dataset.")

        # Extract expression data for marker genes
        expression_data = self.adata[:, self.all_marker_genes].X
        if hasattr(expression_data, "toarray"):
            expression_data = expression_data.toarray()
        expression_df = pd.DataFrame(
            expression_data,
            columns=self.all_marker_genes,
            index=self.adata.obs_names,
        )
        obs_df = self.adata.obs.copy()
        self.cells_df = obs_df.join(expression_df)

        # Prepare data for the dot plot
        dp_results = []
        groupby = self.cells_df[self.groupby]
        for gene in self.all_marker_genes:
            for cluster, indices in groupby.groupby(
                groupby, observed=True
            ).groups.items():
                cluster_data = self.cells_df.loc[indices, gene]
                percentage = (cluster_data > self.expression_cutoff).mean() * 100
                mean_expression = cluster_data.mean()
                dp_results.append(
                    {
                        "gene_id": gene,
                        "cluster": cluster,
                        "percentage": percentage,
                        "mean_expression": mean_expression,
                        # "size": (percentage / 100) * self.max_dot_size,
                    }
                )
        dp_df = pd.DataFrame(dp_results)
        dp_df["cluster"] = dp_df["cluster"].astype(str)
        dp_df["size"] = (
            dp_df["percentage"] / dp_df["percentage"].max()
        ) * self.max_dot_size
        dp_df["mean_expression_norm"] = (
            dp_df["mean_expression"] / dp_df["mean_expression"].max()
        )

        self.data = dp_df

class DotPlotPlot(PointPlot):
    # Just using the pointplot for now.. may need to customize later,
    # e.g. to add a dotplot size/color legend
    pass

hv.Store.register({DotPlot: DotPlotPlot}, backend="bokeh")

options = hv.Store.options(backend="bokeh")
options.DotPlot = hv.Options(
    "style",
    color="mean_expression",
    cmap="Reds",
    size="size",
)
options.DotPlot = hv.Options(
    "plot",
    responsive=True,
    min_height=300,
    tools=["hover"],
    xlabel="Gene",
    ylabel="Cluster",
    xrotation=45,
    colorbar=True,
    colorbar_position="left",
    invert_yaxis=True,
    show_legend=False,
    fontscale=0.7,
    xaxis='top',
)
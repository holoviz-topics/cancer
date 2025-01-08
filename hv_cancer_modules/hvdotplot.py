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
        default=["cluster", "marker_line"],
        bounds=(2, 2),
        doc="Key dimensions representing cluster and marker line (combined marker cluster name and gene).",
    )

    vdims = param.List(
        default=["gene_id", "mean_expression", "size", "percentage", "mean_expression_norm", "marker_cluster_name"],
        doc="Value dimensions representing expression metrics and metadata.",
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
        # Flatten the marker_genes preserving order and duplicates
        all_marker_genes = []
        for marker_cluster_name, genes in self.marker_genes.items():
            for gene in genes:
                all_marker_genes.append(gene)

        # Check if all genes are present in adata.var_names, warn about missing ones
        missing_genes = set(all_marker_genes) - set(self.adata.var_names)
        if missing_genes:
            print(f"Warning: The following genes are not present in the dataset and will be skipped: {missing_genes}")
            all_marker_genes = [g for g in all_marker_genes if g not in missing_genes]
            if not all_marker_genes:
                raise ValueError("None of the specified marker genes are present in the dataset.")

        # Extract expression data for the included marker genes
        expression_data = self.adata[:, all_marker_genes].X
        if hasattr(expression_data, "toarray"):
            expression_data = expression_data.toarray()
        expression_df = pd.DataFrame(expression_data, columns=all_marker_genes, index=self.adata.obs_names)

        obs_df = self.adata.obs.copy()
        self.cells_df = obs_df.join(expression_df)

        dp_results = []
        group_series = self.cells_df[self.groupby]

        for marker_cluster_name, gene_list in self.marker_genes.items():
            for gene in gene_list:
                if gene not in expression_df.columns:
                    continue
                for cluster_value, indices in group_series.groupby(group_series, observed=True).groups.items():
                    cluster_data = self.cells_df.loc[indices, gene].values
                    percentage = (cluster_data > self.expression_cutoff).mean() * 100
                    mean_expression = cluster_data.mean()
                    marker_line = f"{marker_cluster_name}, {gene}"
                    dp_results.append(
                        {
                            "marker_line": marker_line,
                            "cluster": str(cluster_value),
                            "marker_cluster_name": marker_cluster_name,
                            "gene_id": gene,
                            "percentage": percentage,
                            "mean_expression": mean_expression,
                        }
                    )

        dp_df = pd.DataFrame(dp_results)

        if dp_df.empty:
            dp_df = pd.DataFrame(
                columns=["marker_line", "cluster", "marker_cluster_name", "gene_id", "percentage", "mean_expression", "size", "mean_expression_norm"]
            )

        if not dp_df.empty:
            dp_df["size"] = (dp_df["percentage"] / dp_df["percentage"].max()) * self.max_dot_size if dp_df["percentage"].max() > 0 else 0
            dp_df["mean_expression_norm"] = dp_df.groupby("marker_line")["mean_expression"].transform(
                lambda x: x / x.max() if x.max() > 0 else 0
            )

        self.data = dp_df


class DotPlotPlot(PointPlot):
    pass

hv.Store.register({DotPlot: DotPlotPlot}, backend="bokeh")

options = hv.Store.options(backend="bokeh")
options.DotPlot = hv.Options(
    "style",
    color="mean_expression_norm",
    cmap="Reds",
    size="size",
)
options.DotPlot = hv.Options(
    "plot",
    responsive=True,
    min_height=300,
    tools=["hover"],
    ylabel="Cluster",
    xlabel="Marker Cluster, Gene",
    xrotation=45,
    colorbar=True,
    colorbar_position="left",
    invert_yaxis=True,
    show_legend=False,
    fontscale=0.7,
    xaxis='top',
    clabel="Mean expression in group"
)

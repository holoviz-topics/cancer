import panel as pn

pn.extension()

import anndata as ad
import holoviews as hv
import pandas as pd
import numpy as np
import holoviews.operation.datashader as hd
import datashader as ds
import hvplot.pandas  # noqa

from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import panel as pn
import param


pn.extension()
hv.extension("bokeh")


marker_genes = {
    "CD14+ Mono": ["FCN1", "CD14"],
    "CD16+ Mono": ["TCF7L2", "FCGR3A", "LYN"],
    # Note: DMXL2 should be negative
    "cDC2": ["CST3", "COTL1", "LYZ", "DMXL2", "CLEC10A", "FCER1A"],
    "Erythroblast": ["MKI67", "HBA1", "HBB"],
    # Note HBM and GYPA are negative markers
    "Proerythroblast": ["CDK6", "SYNGR1", "HBM", "GYPA"],
    "NK": ["GNLY", "NKG7", "CD247", "FCER1G", "TYROBP", "KLRG1", "FCGR3A"],
    "ILC": ["ID2", "PLCG2", "GNLY", "SYNE1"],
    "Naive CD20+ B": ["MS4A1", "IL4R", "IGHD", "FCRL1", "IGHM"],
    # Note IGHD and IGHM are negative markers
    "B cells": [
        "MS4A1",
        "ITGB1",
        "COL4A4",
        "PRDM1",
        "IRF4",
        "PAX5",
        "BCL11A",
        "BLK",
        "IGHD",
        "IGHM",
    ],
    "Plasma cells": ["MZB1", "HSP90B1", "FNDC3B", "PRDM1", "IGKC", "JCHAIN"],
    # Note PAX5 is a negative marker
    "Plasmablast": ["XBP1", "PRDM1", "PAX5"],
    "CD4+ T": ["CD4", "IL7R", "TRBC2"],
    "CD8+ T": ["CD8A", "CD8B", "GZMK", "GZMA", "CCL5", "GZMB", "GZMH", "GZMA"],
    "T naive": ["LEF1", "CCR7", "TCF7"],
    "pDC": ["GZMB", "IL3RA", "COBLL1", "TCF4"],
}

# adata.write(filename='adata.h5ad')
adata = ad.read_h5ad("adata.h5ad")


umap_df = pd.DataFrame(adata.obsm["X_umap"], columns=["UMAP1", "UMAP2"])
pca_df = pd.DataFrame(
    adata.obsm["X_pca"],
    columns=[f"PCA{1+i}" for i in range(adata.obsm["X_pca"].shape[-1])],
)

obs_df = adata.obs.join(umap_df.set_index(adata.obs.index))
obs_df = obs_df.join(pca_df.set_index(adata.obs.index))
var_df = adata.var.copy()

# Extract expression data for marker genes
sel_genes = marker_genes["CD16+ Mono"]  # ["TCF7L2", "FCGR3A", "LYN"],
expression_df = pd.DataFrame(
    adata[:, sel_genes].X.toarray(), columns=sel_genes, index=adata.obs_names
)


class CellViewer(pn.viewable.Viewer):
    """
    A Panel viewer class for visualizing cell data with UMAP plots and dotplots.
    """

    leiden_res = param.Selector(default="leiden_res_0.50")
    max_dot_size = param.Integer(default=10)

    def __init__(self, adata, obs_df, marker_genes, expression_cutoff=0.1, **params):
        super().__init__(**params)

        self.adata = adata
        self.obs_df = obs_df
        self.marker_genes = marker_genes
        self.expression_cutoff = expression_cutoff
        self.param["leiden_res"].objects = sorted(
            [
                key
                for key in adata.uns.keys()
                if key.startswith("leiden_res") and not key.endswith("colors")
            ]
        )
        # Initialize the dot plot data
        self.dp_data = self._compute_dotplot_data()

        self.main_placeholder = pn.pane.Placeholder(sizing_mode="stretch_both")
        self.template = pn.template.FastListTemplate(
            title="Cell Viewer",
            main=[self.main_placeholder],
            sidebar=[self.param.leiden_res, self.param.max_dot_size],
            collapsed_sidebar=True,
            main_layout=None,
        )

        pn.state.onload(self._load)

    @pn.depends("leiden_res", "max_dot_size", watch=True)
    def _load(self):
        with self.main_placeholder.param.update(loading=True):
            df = self._prepare_dot_plot_data(self.dp_data)

            umap_plot = self._create_umap_points()

            dendro_data, clusters_ordered = self._prepare_dendrogram(df)
            cluster_positions = {
                cluster: pos for pos, cluster in enumerate(clusters_ordered)
            }
            dotplot = self._plot_dotplot(df, clusters_ordered, cluster_positions)
            dendrogram = self._plot_dendrogram(
                dendro_data, clusters_ordered, cluster_positions
            )

            if "gene_group" in df.columns:
                annotations = self._plot_annotations(df)
                # return (base_plot + annotations).opts(hv.opts.Layout(shared_axes=True))
            linked = hv.link_selections(umap_plot + dotplot).cols(1)

            self.main_placeholder.object = linked

    def _compute_dotplot_data(self):
        """
        Compute data required for creating a dot plot, handling markers as a list or dictionary.
        Allows for genes to appear in multiple groups.
        """
        expression = self.adata.X
        groupby = self.adata.obs[self.leiden_res]
        gene_names = self.adata.var_names
        markers = self.marker_genes
        expression_cutoff = self.expression_cutoff

        if not isinstance(expression, csr_matrix):
            expression = csr_matrix(expression)

        gene_name_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        if isinstance(markers, dict):
            # Flatten markers dictionary to get list of (gene, group)
            marker_genes = []
            for group, genes in markers.items():
                for gene in genes:
                    if gene in gene_name_to_idx:
                        marker_genes.append((gene, group))

            marker_indices = [gene_name_to_idx[gene] for gene, group in marker_genes]
            marker_gene_names = [gene for gene, group in marker_genes]
            gene_groups = [group for gene, group in marker_genes]

        elif isinstance(markers, list):
            marker_genes = [gene for gene in markers if gene in gene_name_to_idx]
            marker_genes = list(
                dict.fromkeys(marker_genes)
            )  # Remove duplicates while preserving order
            marker_indices = [gene_name_to_idx[gene] for gene in marker_genes]
            marker_gene_names = marker_genes
            gene_groups = [None] * len(marker_gene_names)
        else:
            raise ValueError("Markers must be a list or a dictionary.")

        groupby = np.array(groupby)
        clusters = np.unique(groupby)

        clusters_series = pd.Series(clusters)
        clusters_numeric = pd.to_numeric(clusters_series, errors="coerce")
        convert_cluster_to_numeric = not clusters_numeric.isnull().any()

        results = []

        for gene_idx, gene_name, gene_group in zip(
            marker_indices, marker_gene_names, gene_groups
        ):
            gene_expression = expression[:, gene_idx]

            gene_expression_binarized = gene_expression.copy()
            gene_expression_binarized.data = (
                gene_expression_binarized.data > expression_cutoff
            ).astype(int)

            for cluster in clusters:
                cluster_mask = groupby == cluster
                cluster_cell_indices = np.where(cluster_mask)[0]
                n_cells_in_cluster = len(cluster_cell_indices)

                X_cluster = gene_expression[cluster_cell_indices]
                X_cluster_binarized = gene_expression_binarized[cluster_cell_indices]

                expressing_cells = X_cluster_binarized.sum()
                percentage = (expressing_cells / n_cells_in_cluster) * 100

                total_expression = X_cluster.sum()
                mean_expression = total_expression / n_cells_in_cluster

                cluster_results = pd.DataFrame(
                    {
                        "gene": [gene_name],
                        "cluster": [cluster],
                        "percentage": [percentage],
                        "mean_expression": [mean_expression],
                        "gene_group": [gene_group],
                    }
                )
                results.append(cluster_results)

        df = pd.concat(results, ignore_index=True)

        if convert_cluster_to_numeric:
            df["cluster"] = pd.to_numeric(df["cluster"])

        return df

    def _create_umap_points(self):
        """Create the UMAP visualization."""
        df = self.obs_df.copy().rename(columns={self.leiden_res: "cluster_pos"})

        # points = hv.Points(df, kdims=["UMAP1", "UMAP2"], vdims=["cluster_pos"]).opts(
        #     color="cluster_pos",
        #     responsive=True,
        # )

        points = df.hvplot.points(
            x="UMAP1",
            y="UMAP2",
            cmap="Category20",
            # datashade=True,
            # aggregator=ds.count_cat("cluster_pos"),
            tools=["box_select"],
            legend=False,
            grid=False,
            height=500,
            responsive=True,
            hover_cols=["cluster_pos"]
        )
        # points = hd.dynspread(points, threshold=0.9, max_px=15)

        # inspector = hd.inspect_points.instance(
        #     streams=[hv.streams.Tap], transform=self._datashade_hover_transform
        # )

        # inspect_selection = inspector(points).opts(
        #     color="black", tools=["hover"], marker="circle", size=8, fill_alpha=0.1
        # )

        # * inspect_selection

        return points

    def _create_umap_labels(self):
        labels = (
            self.obs_df.groupby(self.leiden_res, as_index=False, observed=False)[
                ["UMAP1", "UMAP2"]
            ]
            .mean()
            .hvplot.labels(
                x="UMAP1",
                y="UMAP2",
                text=self.leiden_res,
                text_color="black",
                hover=False,
                responsive=True,
            )
        )

    def _datashade_hover_transform(self, df):
        """Transform data for hover functionality."""
        aggregated_df = df.groupby("sample", observed=False).agg(
            {
                "n_genes_by_counts": ["mean", "median"],
                "total_counts": ["mean", "median"],
                "log1p_total_counts": ["mean", "median"],
                "pct_counts_in_top_50_genes": "std",
                "pct_counts_in_top_100_genes": "std",
                "pct_counts_in_top_200_genes": "std",
                "pct_counts_in_top_500_genes": "std",
                "PCA1": "mean",
            }
        )

        aggregated_df.columns = [
            "_".join(col).strip() for col in aggregated_df.columns.values
        ]
        aggregated_df["UMAP1"] = df["UMAP1"].mean()
        aggregated_df["UMAP2"] = df["UMAP2"].mean()

        aggregated_row = (
            aggregated_df.reset_index().drop("sample", axis=1).mean().rename("row")
        )
        sample_count = df["sample"].value_counts().rename("row")
        leiden_res = df[self.leiden_res].value_counts().rename("row")
        leiden_res = leiden_res.loc[leiden_res != 0]
        leiden_res.index = "cluster_" + leiden_res.index.astype(str)

        return pd.concat([sample_count, leiden_res, aggregated_row]).to_frame().T

    def _prepare_dendrogram(self, df):
        """Prepare dendrogram data and paths for visualization."""
        cluster_gene_matrix = df.pivot_table(
            index="cluster",
            columns="gene_id",
            values="mean_expression",
            fill_value=0,
            observed=False,
        )

        X = cluster_gene_matrix.values
        cluster_dist = pdist(X, metric="euclidean")
        cluster_linkage = linkage(cluster_dist, method="average")
        dendro_data = dendrogram(
            cluster_linkage, labels=cluster_gene_matrix.index, no_plot=True
        )

        return dendro_data, cluster_gene_matrix.index

    def _plot_dendrogram(self, dendro_data, clusters_ordered, cluster_positions):
        """Generate dendrogram plot from dendrogram data."""
        dendro_paths = []
        icoord = np.array(dendro_data["dcoord"])
        dcoord = np.array(dendro_data["icoord"])

        for xs, ys in zip(icoord, dcoord):
            ys_new = [
                (
                    cluster_positions.get(clusters_ordered[int((y - 5.0) / 10.0)], y)
                    if y % 10 == 5.0
                    else y / max(dcoord.flatten()) * (len(clusters_ordered) - 1)
                )
                for y in ys
            ]
            dendro_paths.append(np.column_stack([xs, ys_new]))

        ylim = (len(clusters_ordered) - 0.5, -0.5)
        return hv.Path(dendro_paths, ["Distance", "Cluster"]).opts(
            xlabel="",
            invert_yaxis=False,
            xaxis=None,
            yaxis="right",
            show_frame=False,
            fontscale=0.6,
            tools=["hover"],
            responsive=True,
            max_width=200,
            max_height=300,
            ylim=ylim,
        )

    def _plot_dotplot(self, df, clusters_ordered, cluster_positions):
        """Generate dot plot from prepared data."""
        df["cluster_pos"] = df["cluster"].map(cluster_positions)
        yticks = [(pos, cluster) for pos, cluster in enumerate(clusters_ordered)]
        ylim = (
            max(cluster_positions.values()) + 0.5,
            min(cluster_positions.values()) - 0.5,
        )

        return hv.Points(
            df,
            kdims=["gene_id", "cluster_pos"],
            vdims=[
                "mean_expression_normalized",
                "size",
                "percentage",
                "mean_expression",
                "gene_group",
            ],
        ).opts(
            xrotation=90,
            color="mean_expression_normalized",
            cmap="Reds",
            size="size",
            line_color="black",
            line_alpha=0.1,
            marker="o",
            tools=["hover"],
            colorbar=True,
            colorbar_position="left",
            min_width=850,
            height=300,
            responsive=True,
            xlabel="Gene",
            ylabel="Cluster",
            yticks=yticks,
            ylim=ylim,
            fontscale=0.6,
        )

    def _plot_annotations(self, df):
        """Generate annotations heatmap if gene_group is present."""
        gene_groups = df[["gene_id", "gene_group"]].drop_duplicates()
        gene_groups["group_code"] = gene_groups["gene_group"].factorize()[0]

        annotations_df = gene_groups.assign(Group="Group")
        return hv.HeatMap(
            annotations_df,
            kdims=["gene_id", "Group"],
            vdims=["group_code", "gene_group"],
        ).opts(
            responsive=True,
            colorbar=False,
            xaxis=None,
            yaxis=None,
            cmap=["darkgrey", "lightgrey"],
            tools=["hover"],
            toolbar=None,
            height=50,
            show_frame=False,
            min_width=850,
        )

    def _prepare_dot_plot_data(self, df):
        """Prepare data for dot plot visualization."""
        df["gene_id"] = df.apply(
            lambda row: (
                f"{row['gene']} ({row['gene_group']})"
                if pd.notnull(row["gene_group"])
                else row["gene"]
            ),
            axis=1,
        )

        df["size"] = (df["percentage"] / df["percentage"].max()) * self.max_dot_size
        df["mean_expression_normalized"] = (
            df["mean_expression"] / df["mean_expression"].max()
        )

        gene_ids_order = df["gene_id"].drop_duplicates().tolist()
        df["gene_id"] = pd.Categorical(
            df["gene_id"], categories=gene_ids_order, ordered=True
        )

        return df

    def _create_gene_group_annotations(self, df):
        """Create gene group annotation heatmap."""
        gene_groups = df[["gene_id", "gene_group"]].drop_duplicates()
        gene_groups["group_code"] = gene_groups["gene_group"].factorize()[0]

        annotations_df = gene_groups[["gene_id", "group_code", "gene_group"]].copy()
        annotations_df["Group"] = "Group"

        annotations_df["gene_id"] = pd.Categorical(
            annotations_df["gene_id"],
            categories=df["gene_id"].cat.categories,
            ordered=True,
        )
        annotations_df["Group"] = pd.Categorical(
            annotations_df["Group"], categories=["Group"], ordered=True
        )

        return hv.HeatMap(
            annotations_df,
            kdims=["gene_id", "Group"],
            vdims=["group_code", "gene_group"],
        ).opts(
            colorbar=False,
            xaxis=None,
            yaxis=None,
            responsive=True,
            cmap="glasbey_hv",
            tools=["hover"],
            toolbar=None,
            height=50,
            show_frame=False,
            show_grid=False,
        )

    def _create_cluster_gene_matrix(self, df):
        """Create cluster gene matrix for dendrogram computation."""
        return df.pivot_table(
            index="cluster",
            columns="gene_id",
            values="mean_expression",
            fill_value=0,
            observed=False,
        )

    def __panel__(self):
        return self.template


cv = CellViewer(
    adata,
    obs_df,
    marker_genes,
    leiden_res="leiden_res_0.50",
    max_dot_size=10,
)
cv.show()
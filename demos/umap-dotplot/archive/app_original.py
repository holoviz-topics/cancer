
import panel as pn

pn.extension()

import colorcet
import anndata as ad
import holoviews as hv
import pandas as pd
import numpy as np
import holoviews.operation.datashader as hd
import datashader as ds
from holoviews import link_selections
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
    A Panel viewer class for visualizing cell data with UMAP plots and dot plots.
    """

    leiden_res = param.Selector(default="leiden_res_0.50")
    max_dot_size = param.Integer(default=10)

    def __init__(self, adata, obs_df, marker_genes, expression_cutoff=0.1, **params):
        super().__init__(**params)

        self.adata = adata
        self.obs_df = obs_df
        self.marker_genes = marker_genes
        self.expression_cutoff = expression_cutoff

        # Get all marker genes
        self.all_marker_genes = list(
            set(gene for genes in marker_genes.values() for gene in genes)
        )

        # Extract expression data for marker genes
        expression_data = adata[:, self.all_marker_genes].X
        if isinstance(expression_data, csr_matrix):
            expression_data = expression_data.toarray()
        expression_df = pd.DataFrame(
            expression_data, columns=self.all_marker_genes, index=adata.obs_names
        )

        # Merge obs_df and expression_df
        self.cells_df = obs_df.join(expression_df)

        # Create hv.Dataset
        self.cells_dataset = hv.Dataset(self.cells_df)

        self.param["leiden_res"].objects = sorted(
            [
                key
                for key in adata.uns.keys()
                if key.startswith("leiden_res") and not key.endswith("colors")
            ]
        )

        self.main_placeholder = pn.pane.Placeholder(sizing_mode="stretch_both")
        self.template = pn.template.FastListTemplate(
            title="HoloViz Single Cell Gene Expression Demo",
            main=[self.main_placeholder],
            sidebar=[self.param.leiden_res, self.param.max_dot_size],
            collapsed_sidebar=True,
            main_layout=None,
        )

        pn.state.onload(self._load)

    @pn.depends("leiden_res", "max_dot_size", watch=True)
    def _load(self):
        with self.main_placeholder.param.update(loading=True):
            self._setup_selection()
            umap_points = self._plot_umap_points()
            dot_plot = hv.DynamicMap(
                self._plot_dot_plot, streams=[self.selection_stream]
            ).opts(framewise=True)
            self.main_placeholder.object = pn.Column(umap_points, dot_plot)

    def _setup_selection(self):
        # Apply link_selections
        self.selection_linker = link_selections.instance(unselected_alpha=0.4)

        # Create a stream that triggers when selection changes
        self.selection_stream = hv.streams.Stream.define(
            "Selection", selection_expr=None
        )()

        # Update the stream whenever selection changes
        def selection_callback(*events):
            self.selection_stream.event(
                selection_expr=self.selection_linker.selection_expr
            )

        # Attach the callback to the selection_linker
        self.selection_linker.param.watch(selection_callback, "selection_expr")

    def _plot_umap_points(self):
        umap_points = self.obs_df.hvplot.points(
            x="UMAP1",
            y="UMAP2",
            cmap="Category20",
            datashade=True,
            aggregator=ds.count_cat(self.leiden_res),
            tools=["box_select"],
            legend=False,
            grid=False,
            height=500,
            responsive=True,
        ).opts(
            tools=["box_select", "lasso_select"],
            height=500,
            responsive=True,
            xlabel="UMAP1",
            ylabel="UMAP2",
            fontscale=0.6,
        )
        umap_raster = hd.dynspread(
            umap_points,
            threshold=0.95,
            max_px=15,
        )

        labels_df = self.obs_df.groupby(
            self.leiden_res, as_index=False, observed=False
        )[["UMAP1", "UMAP2"]].mean()

        labels_glow = labels_df.hvplot.labels(
            x="UMAP1",
            y="UMAP2",
            text=self.leiden_res,
            text_color="white",
            hover=False,
            responsive=True,
            text_alpha=1,
            text_font_style="bold",
        )
        labels = labels_df.hvplot.labels(
            x="UMAP1",
            y="UMAP2",
            text=self.leiden_res,
            text_color="black",
            hover=False,
            responsive=True,
        )

        inspector = hd.inspect_points.instance(
            streams=[hv.streams.Tap], transform=self._datashade_hover_transform
        )
        inspect_selection = inspector(umap_raster).opts(
            color="black", tools=["hover"], marker="circle", size=8, fill_alpha=0.1
        )

        linked_umap_points = self.selection_linker(umap_raster)
        return linked_umap_points * labels_glow * labels * inspect_selection

    def _datashade_hover_transform(self, df):
        cols = [
            'index',
            'n_genes_by_counts',
            'total_counts',
            'log1p_total_counts',
            'pct_counts_in_top_50_genes',
            'pct_counts_in_top_100_genes',
            'pct_counts_in_top_200_genes',
            'pct_counts_in_top_500_genes',
            'PCA1',
            'UMAP1',
            'UMAP2'
        ]
        if df.empty:
            return pd.DataFrame(columns=cols)
        out = df.iloc[0].to_frame().T
        return out[cols]

    def _plot_dot_plot(self, selection_expr):
        if selection_expr:
            selected_dataset = self.cells_dataset.select(selection_expr)
        else:
            selected_dataset = self.cells_dataset

        df = selected_dataset.data

        if df.empty:
            df = self.cells_df.copy()

        dp_df = self._prepare_dot_plot_data(df)
        dp_points, cluster_positions = self._plot_dp_points(dp_df)
        cluster_gene_matrix, clusters_ordered, _ = self._prepare_dendrogram_data(dp_df, cluster_positions)

        # Create plots
        layout = hv.Layout([])
        layout += dp_points
        try:
            layout += self._plot_dendrogram(
                cluster_gene_matrix, clusters_ordered, cluster_positions
            ).opts(shared_axes=True)
        except Exception as e:
            print(f"Exception in _plot_dendrogram: {e}")
            layout += hv.Path([])
        layout += self._plot_annotations(dp_df).opts(shared_axes=True)

        return layout.opts(shared_axes=True, framewise=True).cols(2)


    def _prepare_dot_plot_data(self, df):
        gene_names = self.all_marker_genes
        groupby = df[self.leiden_res]
        expression_cutoff = self.expression_cutoff
        gene_name_to_idx = {gene: idx for idx, gene in enumerate(gene_names)}

        markers = self.marker_genes
        if isinstance(markers, dict):
            # Flatten markers dictionary to get list of (gene, group)
            marker_genes = []
            for group, genes in markers.items():
                for gene in genes:
                    if gene in gene_name_to_idx:
                        marker_genes.append((gene, group))

            marker_gene_names = [gene for gene, group in marker_genes]
            gene_groups = [group for gene, group in marker_genes]

        elif isinstance(markers, list):
            marker_genes = [gene for gene in markers if gene in gene_name_to_idx]
            marker_genes = list(
                dict.fromkeys(marker_genes)
            )  # Remove duplicates while preserving order
            marker_gene_names = marker_genes
            gene_groups = [None] * len(marker_gene_names)
        else:
            raise ValueError("Markers must be a list or a dictionary.")

        results = []
        for gene_name, gene_group in zip(gene_names, gene_groups):
            gene_expression = df[gene_name]
            gene_expression_binarized = (gene_expression > expression_cutoff).astype(
                int
            )

            for cluster in groupby.unique():
                cluster_mask = groupby == cluster
                n_cells_in_cluster = cluster_mask.sum()
                if n_cells_in_cluster == 0:
                    continue

                X_cluster = gene_expression[cluster_mask]
                X_cluster_binarized = gene_expression_binarized[cluster_mask]

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

        dp_df = pd.concat(results, ignore_index=True)
        dp_df["gene_id"] = dp_df.apply(
            lambda row: (
                f"{row['gene']} ({row['gene_group']})"
                if pd.notnull(row["gene_group"])
                else row["gene"]
            ),
            axis=1,
        )

        gene_ids_order = dp_df["gene_id"].drop_duplicates().tolist()
        dp_df["gene_id"] = pd.Categorical(
            dp_df["gene_id"], categories=gene_ids_order, ordered=True
        )

        dp_df["size"] = (
            dp_df["percentage"] / dp_df["percentage"].max()
        ) * self.max_dot_size
        dp_df["mean_expression_normalized"] = (
            dp_df["mean_expression"] / dp_df["mean_expression"].max()
        )

        return dp_df

    def _prepare_dendrogram_data(self, dp_df, cluster_positions):
        cluster_gene_matrix = dp_df.pivot_table(
            index="cluster",
            columns="gene_id",
            values="mean_expression",
            fill_value=0,
            observed=False,
        )
        clusters_ordered = sorted(cluster_gene_matrix.index, key=lambda x: int(x))
        return cluster_gene_matrix, clusters_ordered, cluster_positions

    def _plot_dp_points(self, dp_df):
        # Map clusters to positions
        dp_df["cluster"] = dp_df["cluster"].astype(str)
        cluster_labels = dp_df["cluster"].unique()
        cluster_positions = {label: idx for idx, label in enumerate(sorted(cluster_labels, key=int))}
        dp_df["cluster_pos"] = dp_df["cluster"].map(cluster_positions)

        yticks = list(cluster_positions.values())
        ytick_labels = list(cluster_positions.keys())
        ylim = min(yticks) - 0.5, max(yticks) + 0.5

        dp_points = hv.Points(
            dp_df,
            kdims=["gene_id", "cluster_pos"],
            vdims=[
                "mean_expression_normalized",
                "size",
                "percentage",
                "mean_expression",
            ],
        ).opts(
            xrotation=45,
            color="mean_expression_normalized",
            cmap="Reds",
            size="size",
            line_color="black",
            line_alpha=0.1,
            marker="o",
            tools=["hover"],
            colorbar=True,
            colorbar_position="left",
            min_height=400,
            responsive=True,
            xlabel="Gene",
            ylabel="Cluster",
            yticks=list(zip(yticks, ytick_labels)),
            ylim=ylim,
            invert_yaxis=True,
            show_legend=False,
            fontscale=0.7,
            xaxis='top',
        )
        return dp_points, cluster_positions


    def _plot_annotations(self, dp_df):
        gene_groups = dp_df[["gene_id", "gene_group"]].drop_duplicates()
        gene_groups["group_code"] = gene_groups["gene_group"].factorize()[0]

        annotations_df = gene_groups.assign(Group="Group")
        annotations_plot = hv.HeatMap(
            annotations_df,
            kdims=["gene_id", "Group"],
            vdims=["group_code", "gene_group"],
        ).opts(
            responsive=True,
            colorbar=False,
            xaxis=None,
            # yaxis='bare',
            # cmap=['darkgrey', 'lightgrey'],
            color_levels=len(annotations_df.group_code.unique()),
            tools=["hover"],
            toolbar=None,
            height=50,
            show_frame=False,
            min_width=850,
            ylabel = '',
        )
        return annotations_plot

    def _plot_dendrogram(
        self, cluster_gene_matrix, clusters_ordered, cluster_positions
    ):
        X = cluster_gene_matrix.values
        cluster_dist = pdist(X, metric="euclidean")
        cluster_linkage = linkage(cluster_dist, method="average")
        dendro_data = dendrogram(
            cluster_linkage, labels=clusters_ordered, orientation='left', no_plot=True
        )

        dendro_paths = []
        icoord = np.array(dendro_data["dcoord"])
        dcoord = np.array(dendro_data["icoord"])

        # Create a mapping from cluster labels to positions
        pos_dict = {str(k): v for k, v in cluster_positions.items()}

        # Mapping of leaves in dendrogram to cluster positions
        ivl = dendro_data['ivl']
        leaf_positions = [pos_dict[label] for label in ivl]

        # Maximum distance for scaling internal nodes
        max_dcoord = np.max(dcoord)

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

        return hv.Path(dendro_paths, ["distance", "cluster_pos"]).opts(
            xlabel="",
            ylabel="",
            invert_yaxis=True,
            xaxis=None,
            yaxis=None,
            show_frame=False,
            fontscale=0.7,
            tools=[],
            responsive=True,
            width=200,
            line_width=2,
            line_alpha=.6,
            line_color='black',
            # min_height=400,
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
cv.servable()
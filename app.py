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
adata = ad.read_h5ad('adata.h5ad')


umap_df = pd.DataFrame(adata.obsm['X_umap'], columns=['UMAP1', 'UMAP2'])
pca_df = pd.DataFrame(adata.obsm['X_pca'], columns=[f'PCA{1+i}' for i in range(adata.obsm['X_pca'].shape[-1])])

obs_df = adata.obs.join(umap_df.set_index(adata.obs.index))
obs_df =  obs_df.join(pca_df.set_index(adata.obs.index))
var_df = adata.var.copy()

# Extract expression data for marker genes
sel_genes = marker_genes['CD16+ Mono'] #["TCF7L2", "FCGR3A", "LYN"],
expression_df = pd.DataFrame(
    adata[:, sel_genes].X.toarray(), 
    columns=sel_genes, 
    index=adata.obs_names
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
            [key for key in adata.uns.keys() if key.startswith("leiden_res") and not key.endswith("colors")]
        )
        self.reset_button = pn.widgets.Button(name="Reset selection", button_type="primary")
        # Initialize the dot plot data
        self.dp_data = self._compute_dotplot_data()

        self.main_placeholder = pn.pane.Placeholder()
        self.template = pn.template.FastListTemplate(
            title="Cell Viewer",
            main=[self.main_placeholder],
            sidebar=[self.param.leiden_res, self.param.max_dot_size],
            collapsed_sidebar=True,
        )
        
        pn.state.onload(self._load)

    @pn.depends("leiden_res", "max_dot_size", watch=True)
    def _load(self):
        with self.main_placeholder.param.update(loading=True):
            # Set up the selection stream
            self.selection_stream = hv.streams.BoundsXY(bounds=(0,0,0,0))
            pn.bind(self._reset_selection, self.reset_button, watch=True)
            # self.reset_button.on_click(self._reset_selection())
            self.umap_selection_area = hv.DynamicMap(
                self._overlay_selection_area, streams=[self.selection_stream]
            )
            self.umap_plot = self._create_umap_plot() * self.umap_selection_area
            self.dotplot_w_bar = hv.DynamicMap(
                self._plot_dotplot_w_bar, streams=[self.selection_stream]
            )
            self.dotplot_w_dendro = hv.DynamicMap(
                self._plot_dotplot_w_dendro, streams=[self.selection_stream]
            )
            
            self.main_placeholder.object = pn.Column(
                self.reset_button,
                self.umap_plot.opts(active_tools=["box_select"]),
                pn.Tabs(
                    ("DE Dotplot", self.dotplot_w_bar),                    
                    ("MG Dotplot", self.dotplot_w_dendro),
                ),
            )
 
    def _reset_selection(self, event):
        self.selection_stream.reset()
        print('reset', self.selection_stream.bounds)
        self.umap_selection_area.event(bounds=(0,0,0,0))

    def _overlay_selection_area(self, bounds):
        """
        Return visible bounds box as selected area
        """ 
        if bounds is None:
            bounds = (0, 0, 0, 0)
        
        return hv.Bounds(
            bounds, vdims=["y", "x"]
        ).opts(
            alpha=0.1 if bounds else 0,
            line_alpha=0.5,
            line_color="black",
            line_width=1,
            line_dash="dashed",
        )


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

    def _create_umap_plot(self):
        """Create the UMAP visualization."""
        points = self.obs_df.hvplot.points(
            x="UMAP1",
            y="UMAP2",
            cmap="Category20",
            datashade=True,
            aggregator=ds.count_cat(self.leiden_res),
            dynspread=True,
            pixel_ratio=.5,
            tools=["box_select"],
            legend=False,
            grid=False,
            frame_height=500,
            responsive=True,
        )
        self.selection_stream.source = points

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

        inspector = hd.inspect_points.instance(
            streams=[hv.streams.Tap], transform=self._datashade_hover_transform
        )

        inspect_selection = inspector(points).opts(
            color="red", tools=["hover"], marker="square", size=8, fill_alpha=0.1
        )

        return points * labels * inspect_selection

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

    def _plot_dotplot_w_bar(self, bounds):
        """Create dot plot with bar visualization."""
        df = self._get_filtered_data(bounds)

        if len(df) == 0:
            df = self.dp_data.copy()

        df = self._prepare_dot_plot_data(df)

        cluster_values = sorted(self.dp_data["cluster"].unique())
        min_cluster = min(cluster_values)
        max_cluster = max(cluster_values)
        ylim = (max_cluster + 0.5, min_cluster - 0.5)
        yticks = [(cluster, str(cluster)) for cluster in cluster_values]

        points = hv.Points(
            df,
            kdims=["gene_id", "cluster"],
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
            colorbar_position='left',
            # width=600,
            frame_height=300,
            responsive=True,
            xlabel="Gene",
            ylabel="Cluster",
            invert_yaxis=False,
            show_legend=False,
            ylim=ylim,
            yticks=yticks,
            fontscale=0.6,
        )

        if "gene_group" in df.columns:
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

            annotations_heatmap = hv.HeatMap(
                annotations_df,
                kdims=["gene_id", "Group"],
                vdims=["group_code", "gene_group"],
            ).opts(
                responsive=True,
                colorbar=False,
                xaxis=None,
                yaxis=None,
                cmap=['darkgrey', 'lightgrey'] * (len(df["gene_id"].cat.categories)//2),
                tools=["hover"],
                toolbar=None,
                height=50,
                show_frame=False,
                show_grid=False,
            )

            layout = (
                hv.Layout([annotations_heatmap, points])
                .cols(1)
                .opts(hv.opts.Layout(shared_axes=True))
            )
            return layout

        return points

    def _plot_dotplot_w_dendro(self, bounds):
        """Create dot plot with dendrogram visualization."""
        df = self._get_filtered_data(bounds)

        if len(df) == 0:
            df = self.dp_data.copy()
        
        df = self._prepare_dot_plot_data(df)

        # Create cluster gene matrix for hierarchical clustering
        cluster_gene_matrix = df.pivot_table(
            index="cluster",
            columns="gene_id",
            values="mean_expression",
            fill_value=0,
            observed=False,
        )

        # Compute hierarchical clustering
        X = cluster_gene_matrix.values
        cluster_dist = pdist(X, metric="euclidean")
        cluster_linkage = linkage(cluster_dist, method="average")
        dendro_data = dendrogram(
            cluster_linkage, labels=cluster_gene_matrix.index, no_plot=True
        )

        clusters_ordered = dendro_data["ivl"]
        cluster_positions = {
            cluster: pos for pos, cluster in enumerate(clusters_ordered)
        }
        df["cluster_pos"] = df["cluster"].map(cluster_positions)

        # cluster_pos_values = sorted(df["cluster_pos"].unique())
        min_cluster = min(clusters_ordered)
        max_cluster = max(clusters_ordered)
        ylim = (max_cluster + 0.5, min_cluster - 0.5)
        yticks = [(pos, cluster) for pos, cluster in enumerate(clusters_ordered)]
        # yticks = [(clust_pos, str(cluster)) for clust_pos in cluster_pos_values]

        points = hv.Points(
            df,
            kdims=["gene_id", "cluster_pos"],
            vdims=[
                "mean_expression_normalized",
                "size",
                "percentage",
                "mean_expression",
                "gene_group",
            ],
        ).redim(cluster_pos='Cluster').opts(
            xrotation=90,
            color="mean_expression_normalized",
            cmap="Reds",
            size="size",
            line_color="black",
            line_alpha=0.1,
            marker="o",
            tools=["hover"],
            colorbar=True,
            colorbar_position='left',
            frame_height=300,
            responsive=True,
            min_width=700,
            xlabel="Gene",
            ylabel="Cluster",
            invert_yaxis=False,
            show_legend=False,
            yticks=yticks,
            ylim=ylim,
            fontscale=0.6,
        )

        # Create dendrogram paths
        dendro_paths = []
        icoord = np.array(dendro_data["dcoord"])
        dcoord = np.array(dendro_data["icoord"])

        for xs, ys in zip(icoord, dcoord):
            ys_new = []
            for y in ys:
                if y % 10 == 5.0:
                    leaf_id = int((y - 5.0) / 10.0)
                    ys_new.append(cluster_positions[clusters_ordered[leaf_id]])
                else:
                    ys_new.append(
                        y / max(dcoord.flatten()) * (len(clusters_ordered) - 1)
                    )
            dendro_paths.append(np.column_stack([xs, ys_new]))

        dendrogram_plot = hv.Path(dendro_paths, ["Distance", "Cluster"]).opts(
            width=200,
            frame_height=300,
            xlabel="",
            invert_yaxis=False,
            xaxis=None,
            yaxis="right",
            show_frame=False,
            # fontsize={"labels": "8pt"},
            fontscale=0.6,
            tools=["hover"],
            yticks=yticks,
            ylim=ylim,
        )

        return (points + dendrogram_plot)

    def _get_filtered_data(self, bounds):
        """Filter data based on bounds selection."""
        print('bounds', bounds)
        if bounds:
            clusters = (
                self.obs_df.loc[
                    (self.obs_df["UMAP1"].between(bounds[0], bounds[2]))
                    & (self.obs_df["UMAP2"].between(bounds[1], bounds[3])),
                    self.leiden_res,
                ]
                .astype(int)
                .tolist()
            )
            return self.dp_data.loc[self.dp_data["cluster"].isin(clusters)].copy()
        return pd.DataFrame({})

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

    def _create_dot_plot_points(self, df):
        """Create the points element for dot plots."""
        cluster_values = sorted(self.dp_data["cluster"].unique())
        min_cluster = min(cluster_values)
        max_cluster = max(cluster_values)
        ylim = (max_cluster + 0.5, min_cluster - 0.5)
        yticks = [(cluster, str(cluster)) for cluster in cluster_values]

        points = hv.Points(
            df,
            kdims=["gene_id", "cluster"],
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
            frame_height=300,
            responsive=True,
            xlabel="Gene",
            ylabel="Cluster",
            invert_yaxis=False,
            show_legend=False,
            ylim=ylim,
            yticks=yticks,
        )

        return points

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


CellViewer(
    adata,
    obs_df,
    marker_genes,
    leiden_res="leiden_res_0.50",
    max_dot_size=10,
).servable()
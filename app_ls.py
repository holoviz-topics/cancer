import anndata as ad
import holoviews as hv
import pandas as pd
import numpy as np
import holoviews.operation.datashader as hd
import datashader as ds
import hvplot.pandas  # noqa
from holoviews import opts, link_selections, dim
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
    expression_cutoff = param.Number(default=0.1)  # Moved to param for accessibility

    def __init__(self, adata, obs_df, marker_genes, **params):
        super().__init__(**params)

        self.adata = adata
        self.obs_df = obs_df
        self.marker_genes = marker_genes

        # Get all marker genes
        self.all_marker_genes = list(set(gene for genes in marker_genes.values() for gene in genes))

        # Extract expression data for marker genes
        expression_data = adata[:, self.all_marker_genes].X
        if isinstance(expression_data, csr_matrix):
            expression_data = expression_data.toarray()
        expression_df = pd.DataFrame(expression_data, columns=self.all_marker_genes, index=adata.obs_names)

        # Merge obs_df and expression_df
        self.cells_df = obs_df.join(expression_df)

        # Create hv.Dataset
        self.cells_dataset = hv.Dataset(self.cells_df)

        self.param["leiden_res"].objects = sorted(
            [key for key in adata.uns.keys() if key.startswith("leiden_res") and not key.endswith("colors")]
        )

        # self.reset_button = pn.widgets.Button(name="Reset selection", button_type="primary")
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
            # Create umap_points
            self.umap_points = hv.Points(
                self.cells_dataset,
                kdims=['UMAP1', 'UMAP2'],
                vdims=[self.leiden_res] + self.all_marker_genes
            ).opts(
                color=self.leiden_res,
                cmap='Category20',
                tools=['hover', 'box_select', 'lasso_select'],
                size=5,
                height=500,
                responsive=True,
                xlabel='UMAP1',
                ylabel='UMAP2',
                fontscale=0.6,
            )

            # Apply link_selections
            self.selection_linker = link_selections.instance()
            self.linked_umap = self.selection_linker(self.umap_points)

            # Create a stream that triggers when selection changes
            self.selection_stream = hv.streams.Stream.define('Selection', selection_expr=None)()

            # Update the stream whenever selection changes
            def selection_callback(*events):
                self.selection_stream.event(selection_expr=self.selection_linker.selection_expr)

            # Attach the callback to the selection_linker
            self.selection_linker.param.watch(selection_callback, 'selection_expr')

            # Create dynamic dotplot
            self.dynamic_dotplot = hv.DynamicMap(self._create_dotplot, streams=[self.selection_stream])

            self.main_placeholder.object = pn.Column(
                # self.reset_button,
                self.linked_umap,
                self.dynamic_dotplot,
            )

    def _create_dotplot(self, selection_expr):
        # Apply the selection expression to the dataset
        if selection_expr:
            selected_dataset = self.cells_dataset.select(selection_expr)
        else:
            selected_dataset = self.cells_dataset

        df = selected_dataset.data

        if df.empty:
            # Return an empty plot or the dotplot for all data
            df = self.cells_df.copy()

        gene_names = self.all_marker_genes
        groupby = df[self.leiden_res]
        expression_cutoff = self.expression_cutoff

        results = []

        for gene_name in gene_names:
            gene_expression = df[gene_name]
            gene_expression_binarized = (gene_expression > expression_cutoff).astype(int)

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
                        # Include 'gene_group' if needed
                    }
                )
                results.append(cluster_results)

        dp_df = pd.concat(results, ignore_index=True)

        # Prepare data for dot plot visualization
        dp_df["size"] = (dp_df["percentage"] / dp_df["percentage"].max()) * self.max_dot_size
        dp_df["mean_expression_normalized"] = dp_df["mean_expression"] / dp_df["mean_expression"].max()

        # Create the dotplot
        points = hv.Points(
            dp_df,
            kdims=["gene", "cluster"],
            vdims=["mean_expression_normalized", "size", "percentage", "mean_expression"]
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
            frame_height=300,
            responsive=True,
            xlabel="Gene",
            ylabel="Cluster",
            invert_yaxis=False,
            show_legend=False,
            fontscale=0.6,
        )

        return points

    def __panel__(self):
        return self.template



CellViewer(
    adata,
    obs_df,
    marker_genes,
    leiden_res="leiden_res_0.50",
    max_dot_size=10,
).servable()
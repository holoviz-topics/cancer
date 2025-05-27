# Workflow plans

For single cell
* Workflow #1: How to make a figure 1: “cell type validation for cell atlas”
  * Show clusters, do they look ok?
  * Show sample cell type distributions within different patients
  * Look at individual marker set genes across different cell type clusterings
  * TODO: DR to clean up and merge PR
* Workflow #2: How to visualize, explore, (and drill into/select) a very large UMAP
  * TRY TO USE: Tabulo-Sapiens dataset (1.1 million cells)
  * Set Datashader to True in ManifoldMap
  * Explain server side hover tool 
  * Expand later to ‘drill into’ selection to produce something like a distribution plot of the selection or something..
  * There’s a group interested in datashader for there use case
* Workflow #3: ManifoldMap linked to DotMap plot
  * Selection of points in UMAP linked to filtering on the DotMap (with gene selection programmatically specified)
  * Maybe: HoloNote selection of multiple groups to compare on a linked DotMap plot
* Workflow #4: Various plot in HoloViz (Evaluation of a scoring metric across cell types and cells)
  * Charlie will code this out
Bulk 
  * Workflow #5: RNA clustering and expression plotting
  * Workflow #6: extension to include copy number and mutation data

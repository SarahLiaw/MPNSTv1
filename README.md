# MPNSTv1
Sarah Liaw - SURF 2022

Main directory: pipeline.

data_path
- 
- data_transformation: reading the data, transforming X using standard scalars, label
encoders on y (target: *strings* to *floats*).

dim_red
- 
- Dimensionality reduction methods (PCA, UMAP, t-SNE).

model
- 
- logreg1: logistic regression with LASSO penalty using a LOOCV to check.

visualizations
- 
- def_visualizations: functions for visualizations including biplot, heatmap, barchart, variance explained graph etc.
- make_visualizations: running files to create the visualizations.

visuals
- 
- 
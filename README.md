# MPNSTv1
Sarah Liaw - SURF 2022

Main directory: pipeline.

data_path
- 
- data_transformation: reading the data, transforming X using standard scalars, label
encoders on y (target: *strings* to *floats*).

dim_red
- 
- pca: pca reduction.
- tsne
- umap_main: umap reduction (does not currently work due to umap module imports issues with directory.)

model
- 
- logreg1: logistic regression with LASSO penalty using a LOOCV to check.
- adaboost: adaboost with base classifier (80% for some n-estimator).

visualizations
- 
- def_visualizations: functions for visualizations including biplot, heatmap, barchart, variance explained graph etc.
- make_visualizations: running files to create the visualizations.

visuals
- 
- 
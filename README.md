# MPNSTv1
Sarah Liaw - SURF 2022

- Visualization: pipeline.
- DELFI score: delfi_score

delfi_score
- 
- healthy_vs_plexiform: binary classification of healthy and plexiform
using PCA to reduce delfi ratios and z-scores (39) as features.
- mpnst_vs_plexiform: binary classification of mpnst vs plexiform.

METHOD:
- Did 10 iterations of k-fold=5, so there would be 50 values at the end for each library id. 
- For each training set (k-1 folds), I ran PCA over it and took the minimum number of PCs which accounts for 90% of the variance, using explained_variance.
- For the remaining 1 fold, I transformed the PC vectors from the training set to the testing dataset (new basis). 
- I then ran logistic regression with LASSO penalty, which was fitted over the training dataset which has the z-scores and PCs as the features.
- Then I predicted using the model on the testing dataset.
- I saved these values in a dictionary with respect to its library index - this gives 50 values for each library id due to 10 iterations of 5-fold cross validation.
- Then I found the average of these 50 values, which corresponds to the DELFI score of the library index.
- I have also decided to save each training and testing index split so we can go back and check for any errors in the future.
- I repeated the steps for the Healthy vs Plexiform dataset.

delfi_score/data
- 
- arm_frag_rm2otlr (removed outliers, z-score arm frag)
- delfi_diagnosis_rm2otlr (removed outliers, delfi ratios)

delfi_score/results
- 
- mplx_delfi: MPNST(0.0) vs Plexiform(1.0)
- plxhealthy_delfi: healthy(0.0) vs Plexiform(1.0)

delfi_score/abstractions
- 
- Relevant abstractions used in code to reduce duplication.


pipeline/data_path
- 
- data_transformation: reading the data, transforming X using standard scalars, label
encoders on y (target: *strings* to *floats*).

pipeline/dim_red
- 
- pca: pca reduction.
- tsne
- umap_main: umap reduction (does not currently work due to umap module imports issues with directory.)

pipeline/model
- 
- logreg1: logistic regression with LASSO penalty using a LOOCV to check.
- adaboost: adaboost with base classifier (80% for some n-estimator).

pipeline/visualizations
- 
- def_visualizations: functions for visualizations including biplot, heatmap, barchart, variance explained graph etc.
- make_visualizations: running files to create the visualizations.

pipeline/visuals
- 
- Visuals for different sets of data.

pipeline/Relevant Calc
- 
- abstractions: abstractions to calculate DELFI score
- DELFI_score (MPNST vs Plexiform): used to find delfi score for binary
classification of MPNST vs Plexiform (where plexiform=1, mpnst=0).
- DELFI_score (Plexiform vs Healthy): used to find delfi score for binary classification
of Plexiform vs Healthy (where plexiform=1, healthy=0)

pipeline/DELFI score
- 
- Mainly irrelevant scratchwork for Relevant Calc directory.

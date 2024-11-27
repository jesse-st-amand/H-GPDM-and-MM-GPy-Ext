See requirements.txt for module requirements.

To reproduce the results in 'Single-Example Learning in Gaussian Process Dynamical Miture Models', do the following:
run ./model_comparison_core/parallel_sims.py
This will produce results in: ./HGPLVM_output_repository/model_summaries/MCCV/
run ./model_comparison_core/model_results_processing/MCCV_analysis_BM.py
run ./model_comparison_core/model_results_processing/MCCV_analysis_CMU.py
This will produce MCCV significance tests for the bimanual data and CMU data, respectively, in ./HGPLVM_output_repository/MCCV/
Three files per data set are prefixed with:
mccv_analysis_score
mccv_analysis_f1
mccv_analysis_msad
These stand for the frechet distance score, F1 test score, and mean squared acceleration distance, respectively.

Note: GPy mods contains files modified from the GPy Gaussian process package from Sheffield Machine Learning. I have maintained their copyright assertions in the files, but we have no affiliation with their lab or research.
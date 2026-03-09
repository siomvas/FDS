Evaluation Package
==================

Core metrics and gradient-diversity utilities for partition and FL evaluation.

Distribution Metrics
--------------------

.. autosummary::
   :toctree: generated/

   evaluation.metrics.JS_divergence
   evaluation.metrics.TV_label_distributions
   evaluation.metrics.JSD_client_to_global
   evaluation.metrics.FID_distance
   evaluation.metrics.MMD
   evaluation.metrics.energy_distance_features
   evaluation.metrics.sliced_wasserstein_distance
   evaluation.metrics.KS_multivariate_test
   evaluation.metrics.vendi_score
   evaluation.metrics.intrinsic_dimensionality
   evaluation.metrics.knn_separability
   evaluation.metrics.centroid_distance
   evaluation.metrics.feature_variance

Gradient Diversity Metrics
--------------------------

.. autosummary::
   :toctree: generated/

   evaluation.gradient_metrics.GradientDiversityMetric
   evaluation.gradient_metrics.compute_gradient_diversity
   evaluation.gradient_metrics.analyze_gradient_diversity

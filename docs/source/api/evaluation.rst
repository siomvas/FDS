Evaluation Package
==================

.. automodule:: evaluation
   :members:
   :undoc-members:
   :show-inheritance:

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

Label Distribution Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: evaluation.metrics.JS_divergence

.. autofunction:: evaluation.metrics.TV_label_distributions

.. autofunction:: evaluation.metrics.JSD_client_to_global

Feature Distribution Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: evaluation.metrics.FID_distance

.. autofunction:: evaluation.metrics.MMD

.. autofunction:: evaluation.metrics.energy_distance_features

.. autofunction:: evaluation.metrics.sliced_wasserstein_distance

.. autofunction:: evaluation.metrics.KS_multivariate_test

.. autofunction:: evaluation.metrics.vendi_score

.. autofunction:: evaluation.metrics.intrinsic_dimensionality

.. autofunction:: evaluation.metrics.knn_separability

.. autofunction:: evaluation.metrics.centroid_distance

.. autofunction:: evaluation.metrics.feature_variance

Gradient Diversity Metrics
---------------------------

.. autosummary::
   :toctree: generated/

   evaluation.gradient_metrics.GradientDiversityMetric
   evaluation.gradient_metrics.compute_gradient_diversity
   evaluation.gradient_metrics.analyze_gradient_diversity

Gradient Diversity Class
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: evaluation.gradient_metrics.GradientDiversityMetric
   :members:
   :special-members: __init__
   :show-inheritance:

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: evaluation.gradient_metrics.compute_gradient_diversity

.. autofunction:: evaluation.gradient_metrics.analyze_gradient_diversity

Helper Functions
~~~~~~~~~~~~~~~~

.. autofunction:: evaluation.gradient_metrics.flatten_gradients

.. autofunction:: evaluation.gradient_metrics.normalize_gradient

.. autofunction:: evaluation.gradient_metrics.compute_pairwise_cosine_similarity

.. autofunction:: evaluation.gradient_metrics.compute_pairwise_l2_distance

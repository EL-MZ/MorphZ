# API Reference

This page documents the public MorphZ API. Private helpers whose names start
with `_` are intentionally omitted.

## Morph Approximations

```{eval-rst}
.. autoclass:: morphZ.KDEBase
   :members:

.. autoclass:: morphZ.Morph_Indep
   :members: logpdf_kde, resample

.. autoclass:: morphZ.Morph_Pairwise
   :members: logpdf, pdf, resample

.. autoclass:: morphZ.Morph_Group
   :members: logpdf, pdf, resample

.. autoclass:: morphZ.Morph_Tree
   :members: logpdf, resample
```

## Evidence Estimation

```{eval-rst}
.. autofunction:: morphZ.evidence

.. autofunction:: morphZ.bridge_sampling_ln

.. autofunction:: morphZ.compute_bridge_rmse

.. autofunction:: morphZ.bridge_multiprocess.bridge_sampling_ln

.. autofunction:: morphZ.bridge_multiprocess.compute_bridge_rmse
```

## Bandwidth Selection

```{eval-rst}
.. autofunction:: morphZ.bw_method.scott_factor

.. autofunction:: morphZ.bw_method.silverman_factor

.. autofunction:: morphZ.scott_rule

.. autofunction:: morphZ.silverman_rule

.. autofunction:: morphZ.botev_isj_bandwidth

.. autofunction:: morphZ.bw_method.botev_isj_factor

.. autofunction:: morphZ.cross_validation_bandwidth

.. autofunction:: morphZ.select_bandwidth

.. autofunction:: morphZ.compute_and_save_bandwidths
```

## Dependency Analysis

```{eval-rst}
.. autofunction:: morphZ.dependency_tree.compute_pairwise_mi

.. autofunction:: morphZ.dependency_tree.plot_mi_heatmap

.. autofunction:: morphZ.dependency_tree.plot_chow_liu_tree

.. autofunction:: morphZ.dependency_tree.extract_dependency_edges

.. autofunction:: morphZ.dependency_tree.compute_and_plot_mi_tree
```

## Total Correlation

```{eval-rst}
.. autofunction:: morphZ.Nth_TC.stable_seed_for_indices

.. autofunction:: morphZ.Nth_TC.compute_marginal_log_p

.. autofunction:: morphZ.Nth_TC.compute_tc_for_indices

.. autofunction:: morphZ.Nth_TC.compute_total_correlation

.. autofunction:: morphZ.Nth_TC.plot_tc_heatmap

.. autofunction:: morphZ.Nth_TC.compute_and_save_tc
```

## Utilities

```{eval-rst}
.. autofunction:: morphZ.setup_logging

.. autofunction:: morphZ.utils.compute_rho_f2_0_via_statsmodels

.. autofunction:: morphZ.utils.compute_rho_f2_0_via_correlate

.. autofunction:: morphZ.utils.log_plus

.. autofunction:: morphZ.utils.log_sum

.. autofunction:: morphZ.utils.error_bound_from_oscillation
```

Model Factory API
=================

FL model construction is centralized in `fl_training.models.factory`.

Factory
-------

.. automodule:: fl_training.models.factory
   :members:
   :undoc-members:
   :show-inheritance:

Primary entrypoint:

.. autofunction:: fl_training.models.factory.create_model

Model Provider Adapters
-----------------------

Adapters currently route through provider-specific modules under
`fl_training.models.adapters`:

- `timm`
- `smp`
- `hf`
- `torchvision`

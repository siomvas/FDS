Embedding Package
=================

.. automodule:: embedding
   :members:
   :undoc-members:
   :show-inheritance:

Model Registry
--------------

.. autosummary::
   :toctree: generated/

   embedding.ModelRegistry

.. autoclass:: embedding.ModelRegistry
   :members:
   :special-members: __init__
   :show-inheritance:

Embedding Extractors
--------------------

.. autosummary::
   :toctree: generated/

   embedding.EmbeddingExtractor
   embedding.InceptionExtractor
   embedding.DINOv2Extractor
   embedding.CLIPExtractor
   embedding.SigLIPExtractor
   embedding.MAEExtractor

Base Extractor
~~~~~~~~~~~~~~

.. autoclass:: embedding.EmbeddingExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

Pre-trained Extractors
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: embedding.InceptionExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.DINOv2Extractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.CLIPExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.SigLIPExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.MAEExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

Alternative Extractors
~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated/

   embedding.RawPixelResizerExtractor
   embedding.SparseRandomProjectionResizerExtractor
   embedding.UntrainedAutoEncoderExtractor
   embedding.WaveletScatteringExtractor

.. autoclass:: embedding.RawPixelResizerExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.SparseRandomProjectionResizerExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.UntrainedAutoEncoderExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: embedding.WaveletScatteringExtractor
   :members:
   :special-members: __init__
   :show-inheritance:

Cache Manager
-------------

.. autosummary::
   :toctree: generated/

   embedding.CacheManager

.. autoclass:: embedding.CacheManager
   :members:
   :special-members: __init__
   :show-inheritance:

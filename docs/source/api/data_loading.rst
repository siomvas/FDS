Data Loading Package
====================

.. automodule:: data_loading
   :members:
   :undoc-members:
   :show-inheritance:

Dataset Loaders
---------------

.. autosummary::
   :toctree: generated/

   data_loading.dataset.ImageDataset
   data_loading.dataset.ClassificationDataset
   data_loading.dataset.SegmentationDataset
   data_loading.dataset.BaseDataset

Basic Dataset Classes
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data_loading.dataset.ImageDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

.. autoclass:: data_loading.dataset.ClassificationDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

.. autoclass:: data_loading.dataset.SegmentationDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

Registry-Based Loader
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data_loading.dataset.BaseDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

Data Modules
------------

.. autosummary::
   :toctree: generated/

   data_loading.dm.BaseDataModule
   data_loading.dm.FederatedDataset
   data_loading.dm.SegmentationDataset
   data_loading.dm.TinyImageNetDataModule
   data_loading.dm.Food101DataModule
   data_loading.dm.ADE20KDataModule
   data_loading.dm.get_datamodule

Base Data Module
~~~~~~~~~~~~~~~~

.. autoclass:: data_loading.dm.BaseDataModule
   :members:
   :special-members: __init__
   :show-inheritance:

Dataset-Specific Data Modules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: data_loading.dm.TinyImageNetDataModule
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: data_loading.dm.Food101DataModule
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: data_loading.dm.ADE20KDataModule
   :members:
   :special-members: __init__
   :show-inheritance:

Factory Function
~~~~~~~~~~~~~~~~

.. autofunction:: data_loading.dm.get_datamodule

Federated Datasets
~~~~~~~~~~~~~~~~~~

.. autoclass:: data_loading.dm.FederatedDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

.. autoclass:: data_loading.dm.SegmentationDataset
   :members:
   :special-members: __init__, __len__, __getitem__
   :show-inheritance:

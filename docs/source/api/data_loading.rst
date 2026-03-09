Data Loading Package
====================

Core data-loading interfaces for datasets and FL data modules.

Dataset Loaders
---------------

.. autosummary::
   :toctree: generated/

   data_loading.dataset.ImageDataset
   data_loading.dataset.ClassificationDataset
   data_loading.dataset.SegmentationDataset
   data_loading.dataset.DetectionDataset
   data_loading.dataset.BaseDataset

Data Modules
------------

.. autosummary::
   :toctree: generated/

   data_loading.data_modules.BaseDataModule
   data_loading.data_modules.TinyImageNetDataModule
   data_loading.data_modules.Food101DataModule
   data_loading.data_modules.ADE20KDataModule
   data_loading.data_modules.PascalVOCDataModule
   data_loading.data_modules.get_datamodule

"""Import adapters.

Importing this package registers every adapter that ships with the application, so
`get_adapter` and `registered_adapters` see them all.
"""

from anomaly_lab.datasets.adapters import channel_folders as channel_folders
from anomaly_lab.datasets.adapters import csv_table as csv_table
from anomaly_lab.datasets.adapters import folder_classes as folder_classes

__all__ = ["channel_folders", "csv_table", "folder_classes"]

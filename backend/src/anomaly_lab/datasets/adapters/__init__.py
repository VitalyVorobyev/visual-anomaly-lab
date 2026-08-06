"""Import adapters.

Importing this package registers every adapter that ships with the application, so
`get_adapter` and `registered_adapters` see them all.
"""

from anomaly_lab.datasets.adapters import channel_folders as channel_folders

__all__ = ["channel_folders"]

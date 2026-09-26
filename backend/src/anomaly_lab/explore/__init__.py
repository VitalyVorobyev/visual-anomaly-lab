"""Explore: what a frozen encoder sees in one image, asked by clicking on it.

The purpose is intuition, not truth. Nothing here is a result: a map is written to a
bounded scratch directory, served once as an overlay, and evaluated by nothing.

`grid` holds the numpy arithmetic (similarity, k-means, the patch frame) and `store` the
scratch maps and their rendering; both are torch-free, so the API process and the torch-free
CI job use them. `session` is the resident child's side (`jobs/explorer.py`) and imports
torch only inside its functions.
"""

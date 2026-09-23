"""SubspaceAD: the method, and the campaign that chooses how to configure it.

The method is Lendering, Akdag and Bondarev, *SubspaceAD: Training-Free Few-Shot Anomaly
Detection via Subspace Modeling* (CVPR 2026). Frozen patch tokens from a self-supervised
ViT, a PCA subspace fitted to a handful of normal images, and an anomaly score that is the
squared residual orthogonal to that subspace. Nothing is trained.

This package exists because the paper answers its questions for one encoder — DINOv2-G,
which this workbench does not carry — and leaves the ones a deployment actually asks:
which of the six encoders in `dino_backbone.BACKBONES` to spend, at what resolution, and
which blocks to read. The campaign measures those, and a single verdict becomes the
`subspace_ad` plugin's defaults.
"""

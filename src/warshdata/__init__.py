"""Build a Warsh recitation dataset from raw recordings.

Stage 1 (this package, today) is segmentation: cut long recitation recordings at
waqf using ``obadx/recitation-segmenter-v2`` and write one manifest record per
segment.  Labelling and Hugging Face export build on the same manifest.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

"""Dataset acquisition and loading.

schema.py    Source-agnostic ArtworkMetadata / ArtworkRecord.
sources/     Dataset-source-specific implementations (e.g. artic.py).
ingest.py    Generic pipeline: source -> cached images -> palettes -> JSONL.
dataset.py   Typed read-back access to the processed dataset (modeling stage).
"""

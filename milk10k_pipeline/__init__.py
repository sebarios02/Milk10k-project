"""
milk10k_pipeline
=================

Reusable data pipeline for the MILK10k skin lesion classification project.

Modules:
    metadata_analysis  -- metadata field survey + correlation with target
    color_analysis      -- dataset-level color/histogram comparison across classes
    preprocessing        -- single-image and batch preprocessing functions
    data_loader          -- DataLoader that yields (processed_image, label) batches
    visualizer            -- reusable plotting utilities for sanity-checking data
"""

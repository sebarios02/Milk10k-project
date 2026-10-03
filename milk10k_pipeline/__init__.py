"""
milk10k_pipeline
================

Reusable data pipeline for the MILK10k skin lesion classification project.

Session 2 modules (refactored, not rewritten):
    metadata_analysis  -- metadata field survey + association with the target
    color_analysis     -- colour/histogram comparison across classes (+ lesion hue/V, Session 3)
    preprocessing      -- single-image and batch numpy preprocessing (now with strict mode)
    data_loader        -- numpy batch loader (now fails loudly on missing files)
    visualizer         -- plotting utilities (sample grid, class balance, batch summary)
    run_eda            -- Session 2 EDA runner (outputs in eda_outputs/)

Milestone 1 modules:
    config      -- ONE place for paths (env var MILK10K_DATA_DIR), seed, image size, class names
    data        -- load CSVs, image table with dx, lesion table (one row per lesion)
    quality     -- label-consistency checks, missing-value decisions, shortcut cross-tabs
    integrity   -- file existence + Image.verify() + image sizes
    labels      -- label strategy (label_map.json), class weights, sampler weights
    splits      -- split_lesions(): leak-free lesion-level train/val/test
    transforms  -- train_transform / eval_transform (torchvision)
    datasets    -- MILK10kImageDataset, LesionDataset, aggregate_predictions
    loaders     -- DataLoaders, seeding, WeightedRandomSampler, class_weights.json
    reporting   -- DataFrame -> Markdown table helper
"""

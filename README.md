# Deformation-Based Morphometry using Deep Learning

![Workflow](xxx)<!-- Insert image link here -->

This repository contains the source code for the research paper titled "*A deformation-based morphometry framework for disentangling normal aging from Alzheimer's disease using learned normal aging templates*". You can find the paper [here](xxx). <!-- Insert paper link here -->

## TODOs
- [ ] Add requirements
- [ ] Add the link to share the OASIS-3 Dataset
- [ ] Add the link to share the pre-trained weights
- [ ] Add the Jupyter notebook demo

## Requirements

## OASIS-3 Dataset (TBA)
We provide preprocessed OASIS-3 data as part of this study. The data is included in the *npz* format, along with corresponding segmentation masks generated using [SynthSeg](https://github.com/BBillot/SynthSeg) version 2.0.

## Pre-trained Healthy Template Creation Models (TBA)
We offer pretrained Atlas-GAN weights trained on the OASIS-3 Dataset for simulating normal aging. Additionally, we provide inference scripts for extracting the learned diffeomorphic registration module and template generation module.

The extracted healthy templates are also available in NIfTI format, spanning ages from 60 to 90.

## Acknowledgements:
This repository is developed based on the [Atlas-GAN](https://github.com/neel-dey/Atlas-GAN) project and makes extensive use of the [VoxelMorph](https://github.com/voxelmorph/voxelmorph) library. 


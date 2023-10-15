# CPNet: Accurate Contour Preservation for Semantic Segmentation by Mitigating the Impact of Pseudo-boundaries

This repository contains the implementation details of our paper "Accurate Contour Preservation for Semantic Segmentation by Mitigating the Impact of Pseudo-boundaries".

## Source Files

- configs/* : config files
- mmseg/evaluation/metrics/boundary_metric.py: the calculation of boundary IoU
- mmseg/models/decode_heads/\*, mmseg/models/necks/\* , mmseg/models/segmentors/cascade_encoder_decoder_3.py: model files
- mmse/models/losses/*: loss function

## Environment Requirements

- Linux/MacOS/Windows
- Python 3.7+
- CUDA 10.2+
- PyTorch 1.8+
- mmcv 2.0
- mmsegmentation 

Noted: The code is based on [mmsegmentation v1.1.2](https://github.com/open-mmlab/mmsegmentation/tree/v1.1.2). Just follow this guidance [get_started](https://github.com/open-mmlab/mmsegmentation/blob/v1.1.2/docs/en/get_started.md#installation) to install the environment and copy or update the code files of this repository to the corresponding files, and everything will be ready.

## Dataset

- [Vaihingen](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-vaihingen.aspx)
- [Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)

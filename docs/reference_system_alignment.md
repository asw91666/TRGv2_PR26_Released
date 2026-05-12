# Reference System Alignment

## How to get alignment matrix

This document explains how to obtain and utilize the `alignment_matrix` required when evaluating a model trained on **ARKitFace** using the **BIWI** dataset.

To execute the process, `kpt_ind.npy` and `arkit_avg_face.pkl` are required. `arkit_avg_face.pkl` contains the averaged data of all face meshes from the ARKitFace training dataset.

Each data file must be located in `npy/kpt_ind.npy` and `data/arkit_avg_face.pkl`, respectively.

Once prepared, you can calculate the matrix using the following command:
`python find_reference_alignment_matrix.py`

The calculated matrix will be saved as `data/arkit_biwi_align_matrix_{today}.pkl`.

### Download alignment matrix
If you prefer to obtain the alignment matrix without above process, then you can download from [here](https://drive.google.com/file/d/16Uj7UKK9th8NyUmoSY5ey9ODOPqvKfpG/view?usp=sharing).


## How to use alignment matrix

In our training and evaluation protocol, the alignment matrix is used exclusively for the **BIWI** dataset. Therefore, please modify **line 161** of the BIWI data loader to load the generated `data/arkit_biwi_align_matrix_{today}.pkl` file.
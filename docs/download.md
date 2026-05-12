# Datasets

## ARKitFace

You can download the ARKitFace dataset from the following link: [ARKitFace dataset](https://github.com/cbsropenproject/6dof_face).
Follow the instructions in the `Download Dataset` section on the linked page. 
You will be able to obtain both the dataset and the preprocessed files. 
We used the files downloaded from ARKitFace without any modifications.

To run our code, you need the flip index file for training. You can download this file from [here](https://drive.google.com/drive/folders/1Qdj62asVBpj-cGM2mnymD1bqNEBhf6RZ?usp=sharing).

Place the file in the following location: `dataset/ARKitFace/flip_index.npy`.

```
${ROOT}  
|-- dataset  
|   |-- BIWI
|   |-- ARKitFace
|   |   |-- ARKitFace_info_test
|   |   |-- ARKitFace_info_train
|   |   |-- ARKitFace_list
|   |   |-- ARKitFace_resources
|   |   |-- ARKitFace_image_test
|   |   |-- ARKitFace_image_train
|   |   |-- flip_index.npy
|-- ... 
|-- ... 
```

## BIWI

You can download the BIWI dataset from the following link: [BIWI](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database).
To run our code, you need some files for [the predicted bounding box](https://drive.google.com/drive/folders/1Qdj62asVBpj-cGM2mnymD1bqNEBhf6RZ?usp=sharing) and [reference system alignment matrix](https://drive.google.com/file/d/16Uj7UKK9th8NyUmoSY5ey9ODOPqvKfpG/view?usp=sharing). 

Place the downloaded files in the following location:

`dataset/BIWI/download_from_official_site/kinect_head_pose_db/hpdb/annot_mtcnn_fan.pkl`

`data/arkit_biwi_align_matrix_250507`

```
${ROOT}  
|-- data
|   |-- arkit_biwi_align_matrix_250507.pkl
|-- dataset  
|   |-- ARKitFace
|   |-- BIWI
|   |   |-- download_from_official_site
|   |   |   |-- db_annotations
|   |   |   |-- head_pose_masks
|   |   |   |-- kinect_head_pose_db
|   |   |   |   |-- hpdb
|   |   |   |   |   |-- annot_mtcnn_fan.pkl
|   |   |   |   |   |-- 01
|   |   |   |   |   |-- 02
|   |   |   |   |   |-- 03
|   |   |   |   |   |-- ...
|   |   |   |   |   |-- ...
|-- ... 
|-- ... 

```

# Pretrained weights
| Model                   | Train Dataset     | Link                |
| ----------------------- | ----------------- | ------------------- |
| TRGv2                   | ARKitFace         | [Download](https://drive.google.com/drive/folders/1Umj1oJgtuJo0jd25e72DO7DVx7_7rOAi?usp=sharing)        |

```
${ROOT}  
|-- checkpoint
|   |-- TRG_PR26
|   |   |-- checkpoint-30/
|-- ... 
|-- ... 
```

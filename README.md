# PROMA
Matlab source codes of the Probabilistic Rank-One Matrix Analysis (PROMA) algorithm presented in the paper [Probabilistic Rank-One Matrix Analysis with Concurrent Regularization](https://www.ijcai.org/Abstract/16/346).

## Usage
Face recognition with PROMA on 2D images from the FERET dataset: 
```
Demo_PROTA.m
```

## Descriptions of the files in this repository  
 - *DBpart.mat* stores the indices for training (2 samples per class)/test data partition.
 - *FERETC80A45.mat* stores 320 faces (32x32) of 80 subjects (4 samples per class) from the FERET dataset.
 - *Demo_PROMA.m* provides example usage of PROMA for subspace learning and classification on 2D facial images.
 - *PROMA.m* implements the PROMA algorithm.
 - *projPROMA.m* projects matrices into the subspace learned by PROMA.
 - *sortProj.m* sorts features by their Fisher scores in descending order.

## Requirement
[Tensor toolbox v2.6](http://www.tensortoolbox.org/).

## Citation
If you find our codes helpful, please consider cite the following [paper](https://www.ijcai.org/Abstract/16/346):
```
@inproceedings{
    zhou2016PROMA,
    title = {Probabilistic Rank-One Matrix Analysis with Concurrent Regularization},
    author={Yang Zhou and Haiping Lu},
    booktitle = {Proceedings of the International Joint Conference on Artificial Intelligence},
    year = {2016},
    numpages = {7},
    pages = {2428–2434},
    location = {New York, New York, USA},
}
```

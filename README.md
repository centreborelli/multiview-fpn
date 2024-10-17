# Fixed Pattern Noise Removal For Multi-View Single-Sensor Infrared Camera (WACV 2024)

[WACV version  of the article](https://openaccess.thecvf.com/content/WACV2024/papers/Barral_Fixed_Pattern_Noise_Removal_for_Multi-View_Single-Sensor_Infrared_Camera_WACV_2024_paper.pdf)

This repository provides the official implementation of the article **Fixed Pattern Noise Removal For Multi-View Single-Sensor Infrared Camera**.

## How to use

**Offline**
If you have pairs of ground truth data and noisy data, you can use the code as follows:
``` 
python CP/Multi_view_CP_offline.py --refdir 'testset/gt' --indir 'testset/noisy' --outdir './results'
```
If you only have noisy data:
``` 
python CP/Multi_view_CP_offline.py --refdir 'testset/noisy' --outdir './results'
```
To test the code with clean data
``` 
python CP/Multi_view_CP_offline.py --refdir 'testset/noisy' --outdir './results' --add_noise
```

Same for pytorch implementation with Pytorch/Multi_view_GD_offline.py

## DATA
The data is available on the following [link](https://zenodo.org/records/13935908)

## Citation
If you use our work, please cite us with the following:
```
@InProceedings{Barral_2024_WACV,
    author    = {Barral, Arnaud and Arias, Pablo and Davy, Axel},
    title     = {Fixed Pattern Noise Removal for Multi-View Single-Sensor Infrared Camera},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    pages     = {1669-1678}
}
```

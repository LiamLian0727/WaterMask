# WaterMask

This repo is the official implementation of "WaterMask: Instance Segmentation for Underwater Imagery". By Shijie Lian, Hua Li, Runmin Cong, Suqi Li, Wei Zhang, Sam Kwong, and has been accepted by ICCV2023! 🎉🎉🎉

### Requirements
* Python 3.6+
* Pytorch 1.3+
* mmcv-full>=1.3.17, \<1.6.0 (we use mmcv 1.5.3 and mmdetection 2.25.1 in code)

and you need use `pip install -v -e .` to install mmdetection

### Datasets
    data
      ├── UDW
      |   ├── annotations
      │   │   │   ├── train.json
      │   │   │   ├── val.json
      │   ├── train
      │   │   ├── L_1.jpg
      │   │   ├── ......
      │   ├── val

you can get our UIIS dataset in [Baidu Netdisk]() or [Google Drive]() (We'll publish our dataset shortly)

### Training
`python tools/train.py configs/_our_/water_r50_fpn_1x.py --work-dir you_dir_to_save_logs_and_models`

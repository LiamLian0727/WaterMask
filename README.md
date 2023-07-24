# WaterMask

### Requirements
* Python 3.6+
* Pytorch 1.6.0+
* mmcv-full>=1.3.17, \<1.6.0 (we use mmcv 1.5.3 and mmdetection 2.25.1 in code)

### Training
python tools/train.py configs/_our_/water_r50_fpn_1x.py --work-dir you_dir_to_save_logs_and_models

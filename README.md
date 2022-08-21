# equivariant_pcn

Source code of *SO(3) Rotation Equivariant Point Cloud Completion using Attention-based Vector Neurons*.

## Requirements
```
PyTorch 1.8.0
open3d 0.9.0
tensorboardX 2.5
scipy 1.6.1
```

## Prerequisite
Please complile the pytorch ops in the `extension` folder following the readme files, you may also find instructions from [ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), [emd](https://github.com/Colin97/MSN-Point-Cloud-Completion), and [Pointnet2.PyTorch](https://github.com/sshaoshuai/Pointnet2.PyTorch).

## Dataset
Download the [MVP](https://github.com/paul007pl/VRCNet) dataset and put it in the `mvp_dataset` folder, the structure should look like this:

```
├── mvp_dataset
│   ├── mvp_test_gt_2048pts.h5
│   ├── mvp_test_gt_8192pts.h5
│   ├── mvp_test_input.h5
│   ├── mvp_train_gt_2048pts.h5
│   ├── mvp_train_gt_8192pts.h5
│   └── mvp_train_input.h5
```

## Train and test
To train a model from scratch:
```
python train.py  # train with both EMD and CD
## or
python train.py --use_emd False  # train with 2 CDs
```
We also provide pre-trained models, this [model](https://pan.baidu.com/s/17ILNS7-Mb8blwTbwLhQEOQ) (extraction code: d5vm) refer to *ours* in Table 4 of main paper, and this [model_{cd}](https://pan.baidu.com/s/1amz19JYSjKRWkvanCO0aIw) (extraction code: b6j5) refer to *ours_{cd}* in Table 4. Put the whole folder in ```info```. We will also try to upload pre-trained models on google drive.

To train a model using pretrained weights:
```
python train.py --load_dir timefolder_in_info    # e.g. python train.py --load_dir 20220502_091211
```

To test the consistency and accuracy on testset of MVP:
```
python test.py --load_dir timefolder_in_info     # e.g. python test.py --load_dir 20220502_091211
```

## Log file
We provide training log files for tracing and validation, you may need to install [tensorboard](https://www.tensorflow.org/tensorboard/get_started) first, any version should be fine. ```20220531_142345``` stores the *ours* version in Table 4, while ```20220718_150016``` stores the *ours_{cd}* version.
```
cd info/timefolder_in_info/log/
tensorboard --logdir=./
```

Note 1: The scalar *emd_test* in the log file refers to *Y_c^\prime* instead of *Y_c*, please check Section E.1 of Appendix.

Note 2: If you use *ours_{cd}* version, the scalar *emd_test* in the log file refers to CD between *Y_c^\prime* and GT.

## Acknowledgement
We borrow some of code from the following inspiring repositories:
[MSN](https://github.com/Colin97/MSN-Point-Cloud-Completion), 
[ChamferDistancePytorch](https://github.com/ThibaultGROUEIX/ChamferDistancePytorch), 
[Pointnet2](https://github.com/sshaoshuai/Pointnet2.PyTorch), 
[VRCNet](https://github.com/paul007pl/VRCNet), 
[VNN](https://github.com/FlyingGiraffe/vnn).

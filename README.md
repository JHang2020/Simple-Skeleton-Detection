# Simple CodeBase for Skeleton Action Detection Evaluation
This is an repo for skeleton-based action detection, used for the evalution of the self-supervised models, e.g., [HiCLR](https://github.com/JHang2020/HiCLR), [PCM3](https://github.com/JHang2020/PCM3). 

## Dataset
You need to download the PKUMMD untrimmed dataset [here](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).

## Test your model:
You need to choose your model arch and the params, change the config.yaml and run:
```
python main.py detection_ds_linear_evaluation --config /mnt/netdisk/zhangjh/Code/skeleton_detection/detection/config/release/gcn_ntu60/linear_eval/detection.yaml
```

## Citation
If you find this repository useful, please consider citing our paper:
```
@article{zhang2022s,
    title={Hierarchical Consistent Contrastive Learning for Skeleton-Based Action Recognition with Growing Augmentations},
    author={Zhang, Jiahang and Lin, Lilang and Liu, Jiaying},
    journal={arXiv preprint arXiv:2211.13466},
    year={2022},
}
```

```
@inproceedings{zhang2023prompted,
  title={Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning},
  author={Zhang, Jiahang and Lin, Lilang and Liu, Jiaying},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={7175--7183},
  year={2023}
}
```


## Acknowledgement
We sincerely thank the authors for releasing the code of their valuable works. Our code is built based on the PKUMMD.
## Licence
This project is licensed under the terms of the MIT license.
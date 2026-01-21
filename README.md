# FGHFN
Fu jiahao

This is the code for our paper: FGHFN: High-Resolution Fusion Network with Frequency-Domain Guidance for Remote Sensing Semantic Segmentation

## Install


```shell
conda create -n airs python=3.8
conda activate airs
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
需要先将pip版本降低到24.1以下：
pip install pip==24.0 -i https://mirrors.aliyun.com/pypi/simple/
再安装：
pip install -r GeoSeg/requirements.txt
pip install pytorch_wavelets
```


## Data Preprocessing

Download the datasets from the official website and split them yourself.

  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)

Prepare the following folders to organize this repo:

```none
FGHFN
├── GeoSeg (代码)
├── fig_results (save the masks predicted by models)
├── test_log （测试日志）
├── weights (权重目录)
├── data (数据目录)
│   ├── loveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── Test
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
│   ├── potsdam 
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── test (processed)
```



**Vaihingen**

Only the TOP image tiles were used without the DSM and NDSM. And we utilized ID: 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38 for testing, ID: 30 for validation, and the remaining 15 images for training.
```shell
cd FGHFN
```
```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/train_images" \
--mask-dir "data/vaihingen/train_masks" \
--output-img-dir "data/vaihingen/train/images_1024" \
--output-mask-dir "data/vaihingen/train/masks_1024" \
--mode "train" --split-size 1024 --stride 512 
```

Generate the testing set.

```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks_eroded" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded
```

Generate the masks_1024_rgb (RGB format ground truth labels) for visualization.

```
python GeoSeg/tools/vaihingen_patch_split.py \
--img-dir "data/vaihingen/test_images" \
--mask-dir "data/vaihingen/test_masks" \
--output-img-dir "data/vaihingen/test/images_1024" \
--output-mask-dir "data/vaihingen/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt
```

As for the validation set, you can select some images from the training set to build it.



**Potsdam**

We utilized ID: 2_13, 2_14,3_13, 3_14, 4_13, 4_14, 4_15, 5_13, 5_14, 5_15, 6_13, 6_14, 6_15, 7_13 for testing, ID: 2_10 for validation, and the remaining 22 images (except image 7_10 with error annotations) for training.
```shell
cd FGHFN
```
```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/train_images" \
--mask-dir "data/potsdam/train_masks" \
--output-img-dir "data/potsdam/train/images_1024" \
--output-mask-dir "data/potsdam/train/masks_1024" \
--mode "train" --split-size 1024 --stride 1024 --rgb-image 
```

```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks_eroded" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024" \
--mode "val" --split-size 1024 --stride 1024 \
--eroded --rgb-image
```

```shell
python GeoSeg/tools/potsdam_patch_split.py \
--img-dir "data/potsdam/test_images" \
--mask-dir "data/potsdam/test_masks" \
--output-img-dir "data/potsdam/test/images_1024" \
--output-mask-dir "data/potsdam/test/masks_1024_rgb" \
--mode "val" --split-size 1024 --stride 1024 \
--gt --rgb-image
```

**LoveDA**
```shell
cd FGHFN
```
```shell
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Train/Rural/masks_png --output-mask-dir data/loveDA/Train/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Train/Urban/masks_png --output-mask-dir data/loveDA/Train/Urban/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Val/Rural/masks_png --output-mask-dir data/loveDA/Val/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/loveDA/Val/Urban/masks_png --output-mask-dir data/loveDA/Val/Urban/masks_png_convert
```

## Training

"-c" means the path of the config, use different **config** to train different models.
```shell
cd FGHFN
```
```shell
python GeoSeg/train_supervision.py -c GeoSeg/config/potsdam/FGHFN_potsdam.py 
```

```shell
python GeoSeg/train_supervision_dp.py -c GeoSeg/config/vaihingen/FGHFN_vaihingen.py
```

```shell
python GeoSeg/train_supervision_dp.py -c GeoSeg/config/loveda/FGHFN_loveda.py
```

## Testing

我们的训练权重在夸克网盘可下载：https://pan.quark.cn/s/5999a4fe4dd3

Prepare the weights folder to organize this repo:

```none
FGHFN
├── weights (权重目录)
│   ├── loveda
│   │   ├──FGHFN_loveda.ckpt
│   ├── vaihingen
│   │   ├──FGHFN_vaihingen.ckpt
│   ├── potsdam
│   │   ├──FGHFN_potsdam.ckpt
├── ....

```
```shell
cd FGHFN
```

**Potsdam**

```shell
python GeoSeg/test_potsdam.py -c GeoSeg/config/potsdam/FGHFN_potsdam.py -o ~/fig_results/potsdam/FGHFN_potsdam --rgb -t 'd4'
```

**Vaihingen**

```shell
python GeoSeg/test_vaihingen.py -c GeoSeg/config/vaihingen/FGHFN_vaihingen.py -o ~/fig_results/FGHFN_vaihingen/ --rgb -t "d4"
```

**LoveDA** 

```shell
python GeoSeg/test_loveda.py -c GeoSeg/config/loveda/FGHFN_loveda.py -o ~/fig_results/loveda/FGHFN_loveda --rgb --val -t "d4"
```



## Acknowledgement

Our training scripts comes from [GeoSeg](https://github.com/WangLibo1995/GeoSeg). Thanks for the author's open-sourcing code.

- [GeoSeg(UNetFormer)](https://github.com/WangLibo1995/GeoSeg)
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)

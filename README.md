# Sampling-Priors-Augmented Deep Unfolding Network for Robust Video Compressive Sensing

<video src="docs/SPA-DUN_gray1080.mp4"></video>

## 1. Getting Started

### Requirements

- pytorch 1.11.0+cu113
- numpy 1.22.3
- opencv-python 4.6.0
- pillow 9.1.0
- scikit-image 0.19.2
- scipy 1.8.0
- thop 0.1.1
- gdown 4.5.3
- pyyaml 6.0.1

The above package versions are feasible but not unique.

### Training Datasets

Use the provided `download_datasets.py` to structure the training datasets (DAVIS-2017-train/val/test-480p) as our format.

```shell
python download_datasets.py
```

### Testing Datasets

Name | Type | Resolution | Path | Download
-|-|-|-|-
Grayscale Benchmark Datasets | Simulation | 256x256 | Dataset/Simu_test/gray/256 | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)
Largescale Datasets |  Simulation | 1080x1920 | Dataset/Simu_test/gray/1080 | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)
RGB Benchmark Datasets |  Simulation | 512x512 | Dataset/Simu_test/color/512 |  [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)
Largescale RGB Datasets |  Simulation | 1080x1920 | Dataset/Simu_test/color/1080 | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)
Real Captured Datasets | Real | 512x512 | Dataset/Real_test | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)

### Sampling Masks

Type | Path | Download
-|-|-
Simulation | Dataset/Masks/new | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)
Real |  Dataset/Masks/real | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz20vfTQRaXeaUXI?e=hQVh6K) / [Google](https://drive.google.com/drive/folders/1jUkJcbPa1WPxDnY6PLYr4-lmAJ1N_6Ry?usp=sharing)

### Checkpoints

Type | Path | Download
-|-|-
Gray Simu| Checkpoints/SPA-DUN-simu | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz-lC73AILdC0TlA?e=ear9gs) / [Google](https://drive.google.com/drive/folders/1KK-EkQcEqIf5aLZ5NvUNOiAOomOgSkgA?usp=sharing)
Color Simu | Checkpoints/SPA-DUN-color | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz-lC73AILdC0TlA?e=ear9gs) / [Google](https://drive.google.com/drive/folders/1KK-EkQcEqIf5aLZ5NvUNOiAOomOgSkgA?usp=sharing)
Real | Checkpoints/SPA-DUN-real | [OneDrive](https://1drv.ms/f/s!AtMIjuudSyv4iz-lC73AILdC0TlA?e=ear9gs) / [Google](https://drive.google.com/drive/folders/1KK-EkQcEqIf5aLZ5NvUNOiAOomOgSkgA?usp=sharing)

**_Tips: If you are in China, you may need to access OneDrive or GoogleDrive via vpn._**

## 2. Training

Train from scratch following our paper:

```shell
python Model/train.py --useAMP --config gray.yml
```

Or customize the settings by modifying `.yaml` file.

## 3. Inference

### Benchmark Tests (Table 1)

```shell
# Seen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/rand_cr50.mat

# Unseen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/R1_cr50.mat

```
Set the compression ratio by defining `--CR [int]`

Each reconstruction frame will be saved in `Outputs/...` folder.

### RGB Benchmark Tests (Table 2)

```shell
# Seen Pattern
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/color/512  --maskpath Dataset/Masks/new/rand_cr50_512.mat --dir Checkpoints/SPA-DUN-color

# Unseen Pattern
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/color/512  --maskpath Dataset/Masks/new/R1_cr50_512.mat --dir Checkpoints/SPA-DUN-color
```

### Large-scale Tests (Table 3)

```shell
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/gray/1080 --maskpath Dataset/Masks/new/rand_cr50_1080.mat
```

### Large-scale RGB Tests (Table 4)

```shell
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/color/1080 --maskpath Dataset/Masks/new/rand_cr50_1080.mat --dir Checkpoints/SPA-DUN-color
```

### Real Applications

```shell
python Model/test.py --output --CR 10 --real --datapath Dataset/Real_test/cr10 --maskpath Dataset/Masks/real/cr50.mat --dir Checkpoints/SPA-DUN-real
```

### Others

Additional tests can be performed by customizing `--dir`, `--datapath` and `--maskpath`.

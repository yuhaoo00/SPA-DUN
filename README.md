# 1. Getting Started
## 1.1 Requirements
- pytorch 1.11.0+cu113
- numpy 1.22.3
- opencv 4.6.0
- Pillow 9.1.0
- scikit-image 0.19.2
- scipy 1.8.0
- thop 0.1.1
- gdown 4.5.3

The above package versions are feasible but not unique.


## 1.2 Training Datasets
Use the provided `download_datasets.py` to structure the training datasets (DAVIS-2017-train/val/test-480p) as our format.
```shell
python download_datasets.py
```

## 1.3 Testing Datasets
Name | Type | Resolution | Path | Download
-|-|-|-|-
Grayscale Benchmark Datasets | Simulation | 256x256 | Dataset/Simu_test/gray/256 | 
Largescale Datasets |  Simulation | 1080x1920 | Dataset/Simu_test/gray/1080 |
RGB Benchmark Datasets |  Simulation | 512x512 | Dataset/Simu_test/color/512 |  
Largescale RGB Datasets |  Simulation | 1080x1920 | Dataset/Simu_test/color/1080 |
Real Captured Datasets | Real | 512x512 | Dataset/Real_test |


## 1.4 Sampling Masks
Type | Path | Download
-|-|-
Simulation | Dataset/Masks/new | 
Real |  Dataset/Masks/real |

## 1.5 Checkpoints
Type | Path | Download
-|-|-
Gray Simu| Checkpoints/SPA-DUN-simu |
Color Simu | Checkpoints/SPA-DUN-color |
Real | Checkpoints/SPA-DUN-real |

# 2. Training
Train from scratch following our paper:
```shell
python Model/train.py --useAMP --config gray.yml
```

Or customize the settings by modifying `.yaml` file.


# 3. Inference
## 3.1 Benchmark Tests (Table 1)
```shell
# Seen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/rand_cr50.mat

# Unseen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/R1_cr50.mat
```
Set the compression ratio by defining `--CR [int]`

Each reconstruction frame will be saved in `Outputs/...` folder.

## 3.2 Benchmark RGB Tests (Table 2)

```shell
# Seen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/rand_cr50_512.mat --dir Checkpoints/SPA-DUN-color

# Unseen Pattern
python Model/test.py --output --CR 24 --maskpath Dataset/Masks/new/R1_cr50_512.mat --dir Checkpoints/SPA-DUN-color
```

## 3.3 Large-scale Tests (Table 3)
```shell
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/gray/1080 --maskpath Dataset/Masks/new/rand_cr50_1080.mat
```

## 3.4 Large-scale Tests (Table 4)
```shell
python Model/test.py --output --CR 24 --datapath Dataset/Simu_test/gray/1080 --maskpath Dataset/Masks/new/rand_cr50_1080.mat --dir Checkpoints/SPA-DUN-color
```

## 3.5 Real Applications
```shell
python Model/test.py --output --CR 10 --real --datapath Dataset/Real_test/cr10 --maskpath Dataset/Masks/real/cr50.mat --dir Checkpoints/SPA-DUN-real
```

## 3.6 Others
Additional tests can be performed by customizing `--dir`, `--datapath` and `--maskpath`.

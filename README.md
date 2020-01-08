# CV Final Project: Inverse Segmentation

Model URL : 
Demo Video : https://youtu.be/OyJVbbEGA3g

## 模型介紹
改自[SinGAN](https://github.com/tamarott/SinGAN)程式碼，分別使用不同的dataset_size，在每一層金字塔訓練多少次(niter)，有沒有加LayerNorm，和在conv加入dropout

| Model Name   | type     | datasize | niter | LayerNorm | dropout   |
| ------------ | -------- | -------- | ----- | --------- | --------- |
| SinGAN_typeA | SinGAN_C | 4        | 500   | Yes       | No        |
| SinGAN_typeB | SinGAN_C | 4        | 500   | No        | No        |
| SinGAN_typeC | SinGAN_C | 4        | 500   | Yes       | Yes (0.1) |
| SinGAN_typeD | SinGAN_C | 16       | 500   | Yes       | No        |
| SinGAN_typeE | SinGAN_D | 4        | 500   | N/A       | No        |
| SinGAN_typeF | SinGAN_D | 4        | 500   | N/A       | Yes (0.1) |

## 需要以下的套件
建議電腦配備
```
GPU:Geforce RTX2080Ti 11G
OS:Ubuntu 16.04 LTS, 18.04 LTS
cuda版本:10.1
cudnn版本:7.6.5
```
python 所需要的套件
```
matplotlib
scikit-image
scikit-learn
scipy
numpy
torch==1.3.0
torchvision
Pillow
opencv-python==4.1.1.26
imgaug
```
## 如何Training
1. 將目錄移置**Model Name**底下`cd ${your_model_path}`
2. 鍵入指令`python -W ignore seg_train.py`就可以開始訓練，每次訓練都會把Model 放置在 `TrainedModels/cityscapes/scale_factor=0.750000_seg/` 底下，輸出的亂數影像也會在同個目錄。

## 圖片測試
**請使用python3.7執行**
```bash=linux
python running.py --input_picture ${your_seg_image} --singan_model_dir ${your_model_name} --singan_model ${your_trained_model_name}
```
- `${your_trained_model_name}`會放在`./${your_model_name}/TrainedModels/`下
- 建議使用類似city scape的(segmentation, real image) pair 測試

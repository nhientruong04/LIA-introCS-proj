# CNN Model Comparison Pipeline
![](assets/cover.png)

This repository contains the training pipeline for comparing popular Convolutional Neural Network (CNN) models in our report, including AlexNet, VGGNet, YOLO, and ResNet.

**Note: YOLO is not included in this pipeline, please refer to its official repository for training.**

The `train.py` script is used for training a machine learning model. It uses command-line arguments for configuration. Here's how you can use it:

# How to use?
## Training
For training, use the `train.py` as below in your CLI

```bash
python train.py -m <model> -ds <dataset> -bs <batch_size> -e <epochs> --resume <resume> --log <log> --lr <learning_rate> --shuffle <shuffle> --workers <workers>
```

| Option     | Default | Description
| ---------------- | ----------- | ---------- | 
| `-m` , `--model`  | `alexnet` | Choose the model for training. With models having different variants, choose the specific variant to train, e.g `vgg16`, `resnet101` |
| `-ds` , `--dataset`   | `cifar10` | Choose the dataset for training. The value must be `cifar10` or `cifar100` |
| `-bs`, `--batch_size`   | `32` | Set the batch size, must be `integer`
|`-e`, `--epochs`   | `10` | Set the number of epochs, must be `integer`|
|`--resume`   | `''` | Path to the weight for resume training. If specified, the log file of the previous run must also be specified |
|`--log`   | `''` | Name of the log file, must be pickle file with `.pkl` format|
|`--lr`   | `0.001` | Set the learning rate |
|`--shuffle`   | `True` | Shuffle the dataset |
|`--workers`   | `10` | Set the number of workers for the dataloader, must be `integer` |

For example, to train a `ResNet-34`, with a `lr=0.01`, `epoch=20` and log the process in `resnet34.pkl`, we'll use:
```bash
python train.py -m resnet34 -lr 1e-2 -e 20 --log resnet34.pkl
```
The flag `--log` is optional. If not specified, the program will create the log file with different name based on the current time. After the training is done, your result will be stored at `runs` folder and a weight file named `last.pth` will be saved at the current directory.  
To resume training with any model with a prepared weight (`.pt` or `.pth` format), we'll use:
```bash
python train.py -m resnet34 --resume /PATH/TO/YOUR/MODEL/WEIGHT --log resnet34.pkl -e 20 --workers 6
```
*Note: an existed log file can still be used. Passing the name of an existed log file will continue to append results to that file.*
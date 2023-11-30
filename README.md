<div align="center">

<p>
    <a href="https://github.com/nhientruong04/LIA-introCS-proj" target="_blank">
      <img width="100%" src="assets/cover.png" alt="banner" style="border-radius: 22px"></a>
  </p>

# <div align="center">CNN Model Comparison Pipeline</div>

<div>

![GitHub repo size](https://img.shields.io/github/repo-size/nhientruong04/LIA-introCS-proj)
![GitHub issues](https://img.shields.io/github/issues/nhientruong04/LIA-introCS-proj)
![GitHub pull requests](https://img.shields.io/github/issues-pr/nhientruong04/LIA-introCS-proj)
![GitHub](https://img.shields.io/github/license/nhientruong04/LIA-introCS-proj)
![GitHub last commit](https://img.shields.io/github/last-commit/nhientruong04/LIA-introCS-proj)
![GitHub contributors](https://img.shields.io/github/contributors/nhientruong04/LIA-introCS-proj)

</div>

</br>

This repository contains the training pipeline for comparing popular Convolutional Neural Network (CNN) models in our report, including AlexNet, VGGNet, YOLO, and ResNet.

**Note: YOLO is not included in this pipeline, please refer to its [official repository](https://github.com/ultralytics/ultralytics) for training.**

![Prev](assets/preview.png)

</div>

<!-- The `train.py` script is used for training a machine learning model. It uses command-line arguments for configuration. Here's how you can use it: -->

## <div align="center">Documentation</div>

See below for a quickstart installation and usage example. For more detailed documentation, please refer to the [Wiki]() *(coming soon)*.

<details open>
<summary>Clone</summary>

Clone this repository using `git`:

```bash
git clone https://github.com/nhientruong04/LIA-introCS-proj.git
```

[**GitHub CLI**](https://cli.github.com/) is also supported, use the following command instead:

```bash
gh repo clone nhientruong04/LIA-introCS-proj
```

Or you can download the repository as a [**ZIP archive**](https://github.com/nhientruong04/LIA-introCS-proj/archive/refs/heads/master.zip) and extract it.

</details>

<details open>
<summary>Requirements</summary>

Install package and dependencies including all [requirements]() in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install -r requirements.txt
```

For alternative installation methods including [Conda](https://anaconda.org/conda-forge/), [Docker](https://hub.docker.com/), and Git, please refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/).

</details>

### <div align="center">Usage</div>

#### Training

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

## <div align="center">Contribute</div>

Contributions are always welcome! Our project would not be possible without help from our community. Please visit our [contribution guidelines](CONTRIBUTING.md) first. Thank you to all of our contributors!

Shout out to all of the team members who have contributed to this project:

- [**Pham Duc An**]() - 10422002
- [**Tran Hai Duong**]() - 10422021
- [**Vo Thi Hong Ha**]() - 10421015
- [**Nguyen Hoang Anh Khoa**]() - 10422037
- [**Truong Hao Nhien**]() - 10422062
- [**Nguyen Song Thien Phuc**]() - 10422067
- [**Bui Duc Xuan**]() - 10422085

## <div align="center">License</div>

This project is licensed under the [MIT License](LICENSE).

## <div align="center">Contact</div>

For any bug reports and feature requests please visit [GitHub Issues](https://github.com/nhientruong04/LIA-introCS-proj/issues) and contact us via [email](mailto:10422062@student.vgu.edu.vn)

</br>

<div align="center">
    <a href="https://github.com/nhientruong04/LIA-introCS-proj"><img src="https://github.com/ultralytics/assets/raw/main/social/logo-social-github.png" width="3%" alt="LIA GitHub"></a><img src="https://github.com/ultralytics/assets/raw/main/social/logo-transparent.png" width="3%" alt="space">

</div>

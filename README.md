# Deep Learning for Image Classification

The train.py script is used for training a machine learning model. It uses command-line arguments for configuration. Here's how you can use it:

```bash
python train.py -m <model> -ds <dataset> -bs <batch_size> -e <epochs> --resume <resume> --log <log> --lr <learning_rate> --shuffle <shuffle> --workers <workers>
```

- `-m`, `--model`: Choose the model for training. Default is 'alexnet'.
- `-ds`, `--dataset`: Choose the dataset for training. Default is 'cifar10'.
- `-bs`, `--batch_size`: Set the batch size. Default is 32.
- `-e`, `--epochs`: Set the number of epochs. Default is 10.
- `--resume`: Resume training with the given weight file. If specified, the log file of the previous run must also be specified. Default is ''.
- `--log`: Name of the log file. Default is ''.
- `--lr`: Set the learning rate. Default is 1e-3.
- `--shuffle`: Shuffle the dataset. Default is True.
- `--workers`: Set the number of workers for the dataloader. Default is 4.

import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import time
import torchvision
import datetime

import argparse

from dataset import Dataset
import utils

def main(args):
    assert args.log == '' or args.log.endswith('.pkl'), 'Log file must be a pickle file (.pkl).'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training with {device}')

    dataset = Dataset(dataset_name=args.dataset, shuffle=args.shuffle)

    # Get dataset
    settings, train_dataset = dataset.prepare_dataset()
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=settings['train'], num_workers=args.workers)
 
    valid_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=settings['valid'], num_workers=args.workers)
    
    num_classes = settings['num_classes']

    train_model = utils.Model(model_name=args.model, num_classes=num_classes)

    # Get model
    model = train_model.get_model()

    # Load checkpoint if resume given
    if args.resume:
        assert Path(args.resume).exists(), f'The checkpoint file does not exist!'
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        print(f'Resuming training with file {args.resume}')

    model.to(device)

    # Naming log file if no specific name given
    log_file = args.log
    if args.log == '':
        fmt = '%m-%d_%H:%M'
        log_file = f'run_{datetime.datetime.today().strftime(fmt)}.pkl'

    logger = utils.Logger(file_name=log_file, resume=args.resume, device=device, num_classes=num_classes)

    # Loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay = 0.01, momentum = 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, verbose=True)

    print(f'Starting training...\tModel: {args.model}\tDataset: {args.dataset}')
    print(f'Hyper-parameters:\nEpochs_num {args.epochs}\nBatch_size {args.batch_size}\nlr {args.lr}')
    print(f'This run will be logged at {log_file}')

    train(model=model, train_loader=train_loader, valid_loader=valid_loader, criterion=criterion, \
        scheduler=scheduler, optimizer=optimizer, device=device, num_epochs=args.epochs, logger=logger)

def train(model, train_loader, valid_loader, criterion, scheduler, optimizer, device, num_epochs, logger):
    begin_time = time.time()

    print_freq = 50
    total_step = len(train_loader)

    for epoch in range(0, num_epochs):
        train_loss = 0

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            train_loss += loss.item()
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%print_freq==0:
                print(f'Training batch [{i+1}/{total_step}]. Loss: {loss}')
            del images, labels, outputs
            torch.cuda.empty_cache()

        print('Epoch [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, loss.item()))
        
        # Validation
        with torch.no_grad():
            label_list = []
            preds_list = []
            val_loss = 0

            for images, labels in tqdm(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                label_list.append(labels)
                preds_list.append(outputs)

                del images, labels, outputs

        accuracy, prec, rec = logger.log(preds_list, label_list, train_loss, val_loss)

        print(f'Metrics of the network on {len(valid_loader)} validation batches:\t Accuracy: {accuracy*100:.2f}% \t Precision: {prec:.2f} \t Recall:{rec:.2f}') 

        states = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(states, 'last.pth')

        scheduler.step(accuracy)

    end_time = time.time() - begin_time
    print(f'Finished training in {end_time // 60} minutes and {end_time % 60} seconds.')



def parse_arg():
    parser = argparse.ArgumentParser(description="Classification demo",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', help='Choose model for training', type=str, default='alexnet')
    parser.add_argument('-ds', '--dataset', help='Choose dataset for training', type=str, default='cifar10')
    parser.add_argument('-bs', '--batch_size', help='Batch size', type=int, default=32)
    parser.add_argument('-e', '--epochs', help='Num epochs', type=int, default=10)
    parser.add_argument('--resume', help='Resume training with the given weight file, log file of previous run must be specified', type=str, default='')
    parser.add_argument('--log', help='Name of log file', type=str, default='')
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('--shuffle', help='Shuffle dataset', type=bool, default=True)
    parser.add_argument('--workers', help='Num workers for dataloader', type=bool, default=4)
    
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_arg()

    main(args)
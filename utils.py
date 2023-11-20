import torchvision.models as models
from pathlib import Path
from easydict import EasyDict
import pickle
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall
import torch
import os


class Model():
    def __init__(self, model_name, num_classes):
        self.name = model_name
        self.num_classes = num_classes

    def get_model(self):
        model = getattr(models, self.name)(num_classes=self.num_classes)

        return model



class Logger():
    def __init__(self, file_name, resume, device, num_classes) -> None:
        if not os.path.isdir('./runs'):
            os.mkdir('runs')

        log_folder = Path('./runs')
        path = log_folder / file_name

        if resume:
            assert Path(path).exists(), f'{path} does not exist. Please choose the correct path for created log file.'
        else:
            assert not Path(path).exists(), f'File already exist. Change file name.'

            initial_log = EasyDict({'Accuracy': [], 'Precision': [], 'Recall': [], 'Loss_train': [], 'Loss_val': []})
            with open(path, 'wb') as f:
                pickle.dump(initial_log, f)

        self.path = path
        self.resume = resume
                
        self.accuracy_meter = MulticlassAccuracy(num_classes=num_classes).to(device) 
        self.precision_meter = MulticlassPrecision(num_classes=num_classes).to(device)
        self.recall_meter = MulticlassRecall(num_classes=num_classes).to(device)

    def __update(self, accu, prec, rec, train_loss, val_loss):
        log_file = pickle.load(open(self.path, 'rb'))

        #update metrics
        log_file.Accuracy.append(accu)
        log_file.Precision.append(prec)
        log_file.Recall.append(rec)
        log_file.Loss_train.append(train_loss)
        log_file.Loss_val.append(val_loss)

        with open(self.path, 'wb') as f:
            pickle.dump(log_file, f)

    def log(self, preds_list, label_list, train_loss, val_loss):
        '''
        Takes 4 params
            preds_list: list of model outputs (predictions)
            label_list: list of ground truth labels
            train_loss: loss in training phase
            val_loss: loss in valuation phase
        '''

        preds = torch.cat(preds_list, dim=0)
        gt_labels = torch.cat(label_list, dim=0)
        
        accu = self.accuracy_meter(preds, gt_labels)
        prec = self.precision_meter(preds, gt_labels)
        rec = self.recall_meter(preds, gt_labels)

        self.__update(accu=accu, prec=prec, rec=rec, train_loss=train_loss, val_loss=val_loss)

        return (accu, prec, rec)
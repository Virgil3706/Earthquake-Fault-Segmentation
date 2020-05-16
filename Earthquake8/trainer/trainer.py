import importlib

import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker
from functions import *
from model.loss import *
import argparse
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def PR(outputs,labels):
    outputs=outputs.squeeze(1).byte()
    labels=labels.squeeze(1).byte()
    # acc = ((outputs==labels).sum().item() )/ ((outputs==labels).sum().item()+(outputs!=labels).sum().item())
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # TP    predict 和 label 同时为1
    TP += ((outputs == 1) & (labels == 1)).sum().item()
    # TN    predict 和 label 同时为0
    TN += ((outputs == 0) & (labels == 0)).sum().item()
    # FN    predict 0 label 1
    FN += ((outputs == 0) & (labels == 1)).sum().item()
    # FP    predict 1 label 0
    FP += ((outputs == 1) & (labels == 0)).sum().item()
    precision =TP/(TP+FP)
    recall = TP/(TP+FN)
    return TP,FP,TN,FN,precision,recall



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)


    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
    def adjust_learning_rate(optimizer, steps, step_size, gamma=0.1, logger=None):
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * gamma
                if logger:
                    logger.info('%s: %s' % (param_group['name'], param_group['lr']))

    def _train_epoch(self, epoch):
        # writer = torch.utils.tensorboard.SummaryWriter('saved/log')
        self.model.train()
        self.train_metrics.reset()
        ##参数
        modelNo=self.modelNo

        # print("modelNo",modelNo)
        best_iou_threshold = 0.5

        for batch_idx, (images, masks) in enumerate(self.data_loader):

            images = Variable(images.to(device))
            masks = Variable(masks.to(device))
            data = images
            target = masks
            if modelNo!=6:
                outputs = self.model(images)
            else:
                outputs = self.model(images.repeat(1,3,1,1))

            loss = torch.zeros(1).to(device)
            y_preds = outputs
            bceloss = nn.BCELoss()

            if modelNo == 0 or modelNo == 1 or modelNo == 4:
                #             print("bceloss")

                loss = self.criterion(outputs, masks)
            #             loss = cross_entropy_loss_HED(outputs, masks)
            #             loss = nn.BCEWithLogitsLoss(outputs, masks)
            elif modelNo == 2:
                for o in range(5):
                    loss = loss + self.criterion(outputs[o], masks)
                loss = loss + bceloss(outputs[-1], masks)
                y_preds = outputs[-1]
            elif modelNo == 3:
                for o in outputs:
                    loss = loss + self.criterion(o, masks)
                y_preds = outputs[-1]

            elif modelNo==6:
                #args = argparse.ArgumentParser(description='earthquakenew')
                params_dict = dict(self.model.named_parameters())
                base_lr = 5e-8
                weight_decay = 0.0002
                #logger = args.logger
                params = []
                for key, v in params_dict.items():
                    #print("in the loop")
                    if re.match(r'conv[1-5]_[1-3]_down', key):
                        #print("in the condition")
                        if 'weight' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.1, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.2, 'weight_decay': weight_decay * 0, 'name': key}]
                    elif re.match(r'.*conv[1-4]_[1-3]', key):
                        if 'weight' in key:
                            params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
                    elif re.match(r'.*conv5_[1-3]', key):
                        if 'weight' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 100, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 200, 'weight_decay': weight_decay * 0, 'name': key}]
                    elif re.match(r'score_dsn[1-5]', key):
                        if 'weight' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.01, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.02, 'weight_decay': weight_decay * 0, 'name': key}]
                    elif re.match(r'upsample_[248](_5)?', key):
                        if 'weight' in key:
                            params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
                        elif 'bias' in key:
                            params += [{'params': v, 'lr': base_lr * 0, 'weight_decay': weight_decay * 0, 'name': key}]
                    elif re.match(r'.*msblock[1-5]_[1-3]\.conv', key):
                        if 'weight' in key:
                            params += [{'params': v, 'lr': base_lr * 1, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [{'params': v, 'lr': base_lr * 2, 'weight_decay': weight_decay * 0, 'name': key}]
                    else:
                        if 'weight' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.001, 'weight_decay': weight_decay * 1, 'name': key}]
                        elif 'bias' in key:
                            params += [
                                {'params': v, 'lr': base_lr * 0.002, 'weight_decay': weight_decay * 0, 'name': key}]
                #optimizer = torch.optim.SGD(params, momentum=args.momentum,
                #                            lr=args.base_lr, weight_decay=args.weight_decay)
                #loss=cross_entropy_loss2d(outputs[o],masks,cuda,)
                for o in range(10):
                    if modelNo == 6:
                        loss=loss+0.5*self.criterion(outputs[o],masks)/64
                    else:
                        loss=loss+0.5*self.criterion(outputs[o],masks)
                if modelNo == 6:
                    loss=loss+1.1*self.criterion(outputs[-1],masks)/64
                else:
                    loss=loss+1.1*self.criterion(outputs[-1],masks)
                if modelNo == 6:
                    y_preds=F.sigmoid(outputs[-1])
                else:
                    y_preds=outputs[-1]
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            predicted_mask = y_preds > best_iou_threshold



            self.writer.set_step((epoch - 1) * self.len_epoch )
            self.train_metrics.update('loss', loss.item())
            # log pr_curve

            # self.writer.add_pr_curve('precision_recall_curve', 1, masks, y_preds)
            # writer.close()
            # print("loss\n",loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(predicted_mask, masks))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                # self.writer.add_image('input', make_grid(data.to(device), nrow=8, normalize=True))
                # self.writer.add_pr_curve(tag = 'precision_recall_curve',labels=masks,predictions=predicted_mask)
                if modelNo != 4:
                    self.writer.add_image('input', make_grid(data.to(device), nrow=8, normalize=True))
                    self.writer.add_pr_curve('precision_recall_curve', 1, masks, y_preds)

            # a,b,c,d,e,f=PR(predicted_mask, masks)
            # self.writer.add_pr_curve_raw('precision_recall_curve1',0,torch.tensor(a),torch.tensor(b),torch.tensor(c),torch.tensor(d),torch.tensor(e),torch.tensor(f))
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log,val_loss = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step(val_loss)



        return log

    def _valid_epoch(self,epoch):
        val_losses = []


        modelNo = self.modelNo
        best_iou_threshold = 0.5

        bceloss = nn.BCELoss()
        self.model.eval()
        self.valid_metrics.reset()
        for batch_idx, (images, masks) in enumerate(self.valid_data_loader):
            #images, masks = images.to(self.device), masks.to(self.device)
            images = Variable(images.to(device))
            masks = Variable(masks.to(device))
            data = images
            target = masks

            outputs = self.model(data)
            loss = torch.zeros(1).to(device)
            y_preds = outputs
            if modelNo == 0 or modelNo == 1 or modelNo == 4:
                #             print("bceloss")

                loss = self.criterion(outputs, masks)
            #             loss = cross_entropy_loss_HED(outputs, masks)
            #             loss = nn.BCEWithLogitsLoss(outputs, masks)
            elif modelNo == 2:
                for o in range(5):
                    loss = loss + self.criterion(outputs[o], masks)
                loss = loss + bceloss(outputs[-1], masks)
                y_preds = outputs[-1]
            elif modelNo == 3:
                for o in outputs:
                    loss = loss + self.criterion(o, masks)
                y_preds = outputs[-1]
            # print("val_loss\n",loss.data)
            elif modelNo==6:
                for o in range(10):
                    loss = loss + 0.5*self.criterion(outputs[o], masks)
                loss = loss + 1.1*self.criterion(outputs[-1], masks)
                y_preds = outputs[-1]
            val_losses.append(loss.data)
            predicted_mask = y_preds > best_iou_threshold


            self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
            self.valid_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(predicted_mask, masks))
            #10000 preserved for GAN
            if modelNo != 10000:
                self.writer.add_image('input', make_grid(data.to(device), nrow=8, normalize=True))
                self.writer.add_pr_curve('precision_recall_curve', 1, masks, y_preds)
            # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result(),torch.mean(torch.stack(val_losses))

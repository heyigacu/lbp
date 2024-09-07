import torch
import numpy as np

def de_onehot(labels):
    return np.argmax(labels)

def save_predictions_txt(val_dataset, predictions_all, labels_all, filepath, ):
    with open(filepath, 'w') as f:
        for i,batch_preds in enumerate(predictions_all):
            batch_labels = labels_all[i]
            for j,preds in enumerate(batch_preds):
                preds = preds.numpy()
                labels = batch_labels[j]
                if val_dataset.n_tasks == 1:
                    preds = preds*val_dataset.std.numpy()[0]+val_dataset.mean.numpy()[0]
                    f.write(f'{preds.tolist()}, {labels[0]}\n')
                else:
                    f.write(f'{preds.tolist()}, {int(de_onehot(labels.numpy()))}\n')

class FinetuneTrainer():
    def __init__(self, optimizer, lr_scheduler, loss_fn, evaluator, result_tracker, summary_writer, device, label_mean=None, label_std=None, ddp=False, local_rank=0):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.evaluator = evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.label_mean = label_mean
        self.label_std = label_std
        self.ddp = ddp
        self.local_rank = local_rank
            
    def _forward_epoch(self, model, batched_data):
        (smiles, g, ecfp, md, labels) = batched_data
        ecfp = ecfp.to(self.device)
        md = md.to(self.device)
        g = g.to(self.device)
        labels = labels.to(self.device)
        predictions = model.forward_tune(g, ecfp, md)
        return predictions, labels

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        for batch_idx, batched_data in enumerate(train_loader):
            self.optimizer.zero_grad()
            predictions, labels = self._forward_epoch(model, batched_data)
            is_labeled = (~torch.isnan(labels)).to(torch.float32)
            labels = torch.nan_to_num(labels)
            if (self.label_mean is not None) and (self.label_std is not None):
                labels = (labels - self.label_mean)/self.label_std
            loss = (self.loss_fn(predictions, labels) * is_labeled).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.lr_scheduler.step()
            if self.summary_writer is not None:
                self.summary_writer.add_scalar('Loss/train', loss, (epoch_idx-1)*len(train_loader)+batch_idx+1)


    def fit(self, model, train_loader, val_loader, val_dataset, model_save_path, predictions_save_path, n_epochs):
        best_val_result = self.result_tracker.init()
        best_epoch = 0
        for epoch in range(1, n_epochs+1):
            print('epoch', epoch)
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                val_result, labels_all, predictions_all = self.eval(model, val_loader)
                if self.result_tracker.update(np.mean(best_val_result), np.mean(val_result)):
                    best_val_result = val_result
                    best_epoch = epoch
                    self.save_predictions(val_dataset, predictions_all, labels_all, predictions_save_path)
                    self.save_model(model, model_save_path)
                if epoch - best_epoch >= 20:
                    break
        return best_val_result, best_epoch
    
    def fit_all(self, model, train_loader, model_save_path, n_epochs):
        for epoch in range(1, n_epochs+1):
            print('epoch', epoch)
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            self.train_epoch(model, train_loader, epoch)
        self.save_model(model, model_save_path)

    
    def eval(self, model, dataloader):
        model.eval()
        predictions_all = []
        labels_all = []
        for batched_data in dataloader:
            predictions, labels = self._forward_epoch(model, batched_data)
            predictions_all.append(predictions.detach().cpu())
            labels_all.append(labels.detach().cpu())
        result = self.evaluator.eval(torch.cat(labels_all), torch.cat(predictions_all))
        return result, labels_all, predictions_all
    
    def save_model(self, model, save_path):
        torch.save(model.state_dict(), save_path)

    def save_predictions(self, val_dataset, predictions_all, labels_all, save_path):
        save_predictions_txt(val_dataset, predictions_all, labels_all, save_path)
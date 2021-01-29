import os
import random
import logging
from functools import partial

import numpy as np
import torch
from tfrecord.torch.dataset import TFRecordDataset
from torch.utils.tensorboard import SummaryWriter

from src.dataset.segmentation_dataset_torch import preprocess
from src.model.unet_torch import Unet 
from src.training.segmentation_session_arg_parser import SegmentationSessionArgParser

LOGGER = logging.getLogger(__name__)


class SegmentationSessionTorch:
    """Responsible for segmentation model setup and configuration."""

    def __init__(self, args):
        self.args = args

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def run(self):
        self.seed_generators()
        self.create_directories()
        self.load_data()
        self.create_model()
        self.compile_model()

        # freeze encoder
        epochs_elapsed = self.train(self.args.epochs_frozen)

        # unfreeze encoder
        for param in self.model.parameters():
            param.requires_grad = True 

        self.train(
            self.args.epochs_unfrozen + epochs_elapsed, 
            initial_epoch=epochs_elapsed
        )
    
    def seed_generators(self):
        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)

    def create_directories(self):
        os.makedirs(self.args.log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(self.args.log_dir, 'metrics.csv')
        metrics = [
            'epoch',
            'binary_accuracy',
            'loss',
            'val_binary_accuracy',
            'val_loss'
        ]

        with open(self.metrics_file, 'w') as f:
            f.write(','.join(metrics) + '\n')

    def load_data(self):

        train_dataset = TFRecordDataset(
            self.args.train_records,
            index_path=None,
            shuffle_queue_size=self.args.shuffle_buffer_size,
            transform=partial(
                preprocess,
                one_image_label=self.args.one_image_label,
                one_pixel_mask=self.args.one_pixel_mask
            )
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
        )

        if self.args.val_records:
            val_dataset = TFRecordDataset(
                self.args.val_records,
                index_path=None,
                shuffle_queue_size=0, # don't shuffle during validation
                transform=partial(
                    preprocess,
                    one_image_label=self.args.one_image_label,
                    one_pixel_mask=self.args.one_pixel_mask
                )
            )

            self.val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.args.batch_size,
            )
        else:
            self.val_loader = None

    def create_model(self):

        self.model = Unet(
            encoder_weights=self.args.encoder_weights,
            encoder_freeze=True,
            one_image_label=self.args.one_image_label,
            device=self.device
        ).to(self.device)

    def compile_model(self):
        self.loss = torch.nn.BCELoss(reduction='none')
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            eps=1e-7
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            verbose=True
        )

    def train(self, epochs, initial_epoch=0):

        stopping_ct = 0
        stopping_loss = None

        tb_writer = SummaryWriter(
            log_dir=self.args.log_dir
        )

        for epoch in range(initial_epoch, epochs):

            train_losses = []
            train_accs = []
            val_losses = []
            val_accs = []

            self.model.train()
            train_bs = []
            for batch in self.train_loader:

                self.optimizer.zero_grad()
                train_loss, train_acc = self._loss(batch)
                train_loss.backward()
                self.optimizer.step()

                train_losses.append(train_loss.item())
                train_accs.append(train_acc)
                train_bs.append(batch[1].shape[0])

            self.model.eval()
            val_bs = []
            with torch.no_grad():
                for i, batch in enumerate(self.val_loader):

                    if i == self.args.validation_steps:
                        break

                    loss, acc = self._loss(batch)
                    val_losses.append(loss.item())
                    val_accs.append(acc)
                    val_bs.append(batch[1].shape[0])
            
            train_loss = np.average(train_losses, weights=train_bs)
            train_acc = np.average(train_accs, weights=train_bs)
            val_loss = np.average(val_losses, weights=val_bs)
            val_acc = np.average(val_accs, weights=val_bs)

            self.scheduler.step(val_loss)

            metrics = [
                epoch,
                train_acc,
                train_loss,
                val_acc,
                val_loss
            ]
            with open(self.metrics_file, 'a') as f:
                f.write(','.join([str(m) for m in metrics]) + '\n')                 

            print(
                f'Epoch {epoch}, ' +
                f'Train Loss: {round(train_loss, 10)}',
                f'Train Acc: {round(train_acc, 10)}',
                f'Val Loss: {round(val_loss, 10)}',
                f'Val Acc: {round(val_acc, 10)}'
            )

            tb_writer.add_scalar('Loss/train', train_loss, epoch)
            tb_writer.add_scalar('Loss/validation', val_loss, epoch)
            tb_writer.add_scalar('Accuracy/train', train_acc, epoch)
            tb_writer.add_scalar('Accuracy/validation', val_acc, epoch)

            if stopping_loss is None:
                stopping_loss = val_loss
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.args.log_dir, 'model.pth')
                )
            elif stopping_loss <= val_loss:
                stopping_ct += 1
            else:
                stopping_loss = val_loss
                stopping_ct = 0
                torch.save(
                    self.model.state_dict(), 
                    os.path.join(self.args.log_dir, 'model.pth')
                )
            
            if stopping_ct == self.args.patience:
                print(f'Stopping training early - no improvement for {self.args.patience} epochs')
                break

        return epoch + 1

    def _loss(self, batch):
        inputs = batch[0].to(self.device)
        labels = batch[1].to(self.device)
        outputs = self.model(inputs)
        loss = self.loss(outputs, labels)

        preds = torch.round(outputs)
        acc = (preds == labels)

        if len(batch) == 3:
            loss = loss * batch[2].to(self.device)
            acc = acc * batch[2].to(self.device)

        loss = loss.mean()
        acc = acc.sum().item() / labels.shape[0]

        if self.args.one_pixel_mask:
            loss = loss * (labels.shape[2] ** 2)
        else:
            if not self.args.one_image_label:
                acc = acc / (labels.shape[2] ** 2)

        return loss, acc

if __name__ == "__main__":
    args = SegmentationSessionArgParser().parse_args()
    session = SegmentationSessionTorch(args)
    session.run()
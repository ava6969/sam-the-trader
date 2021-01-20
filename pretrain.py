import torch.nn.functional as F
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
from datetime import time
import pytz
from torch.utils.data import DataLoader

from ptsdae.sdae import StackedDenoisingAutoEncoder
import ptsdae.model as ae
from ptsdae.utils import *
from downloader import load_dataset
import numpy as np

NY = 'America/New_York'
import pandas as pd

pt_lowest_val = np.inf
t_lowest_val = np.inf


def pt_epoch_callback(epoch, autoencoder, model_idx, loss_value, validation_loss_value):
    global pt_lowest_val
    if validation_loss_value < pt_lowest_val:
        pt_lowest_val = validation_loss_value
        torch.save(autoencoder.state_dict(), f'pretrained_models/sdae_pt_{epoch}_{model_idx}')
        print('saved model at epoch', epoch, 'with val loss', validation_loss_value)


def epoch_callback(epoch, autoencoder, model_idx, loss_value, validation_loss_value):
    global t_lowest_val
    if validation_loss_value < t_lowest_val:
        t_lowest_val = validation_loss_value
        torch.save(autoencoder.state_dict(), f'trained_models/sdae_t_{epoch}_{model_idx}')
        print('saved model at epoch', epoch, 'with val loss', validation_loss_value)


def clean_up(df_dict, ticker):
    train_data = df_dict[ticker].drop('tic', axis=1)
    train_data = train_data[
        ((train_data.index.time >= time(hour=9, minute=30, second=0, tzinfo=pytz.timezone(NY))) &
         (train_data.index.time <= time(hour=11, minute=30, second=0, tzinfo=pytz.timezone(NY)))) |
        ((train_data.index.time >= time(hour=13, tzinfo=pytz.timezone(NY))) &
         (train_data.index.time <= time(hour=15, tzinfo=pytz.timezone(NY))))]
    train_data = train_data.sort_index()
    train_data = train_data.reset_index(drop=True)
    return train_data


if __name__ == '__main__':
    TICKERS = ['AAPL', 'IBM', 'MSFT', 'NVDA', 'ZM', 'AMZN', 'NDAQ', 'DOW' ]
    TECH_IND = 'MACD!macd MA EMA ATR ROC'
    res = 'minute'
    df_dict = load_dataset(TICKERS, TECH_IND, res)

    database = pd.DataFrame()
    for t in TICKERS:
        df = clean_up(df_dict, t)
        database = database.append(df, True)

    np_df = database.to_numpy()
    cuda = torch.cuda.is_available()
    load_pretrained = True
    batch_size = 256
    pretrain_epochs = 50
    finetune_epochs = 500
    testing_mode = False
    sae_dim = [np_df.shape[1], 10, 16]
    train_data = torch.tensor(np_df).float() if not cuda else torch.tensor(np_df).float().cuda()
    size = train_data.shape[0]
    train_sz = int(size * 0.7)
    val_sz = int(size * 0.2)
    n_workers = 0
    test_sz = size - train_sz - val_sz
    print(f'Train size: {train_sz}, Validation Size: {val_sz}, Test Size: {test_sz}')
    ds_train, ds_val, ds_test = torch.split(train_data, [train_sz, val_sz, test_sz], dim=0)
    print(f'Train shape: {ds_train.shape}, Validation Shape: {ds_val.shape}, Test Shape: {ds_test.shape}')
    autoencoder = StackedDenoisingAutoEncoder(sae_dim, final_activation=None)

    if cuda:
        autoencoder.cuda()
    print("Pretraining stage.")
    if load_pretrained:
        autoencoder = ae.load(autoencoder)
    else:
        ae.pretrain(
            ds_train,
            autoencoder,
            cuda=cuda,
            validation=ds_val,
            epochs=pretrain_epochs,
            batch_size=batch_size,
            optimizer=lambda model: Adam(model.parameters()),
            epoch_callback=pt_epoch_callback,
            num_workers=n_workers
        )
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope
    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss, },
            epoch,
        )

    print("Training stage.")
    ae_optimizer = Adam(autoencoder.parameters())
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        model_index=-1,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        num_workers=n_workers,
        update_callback=training_callback,
        epoch_callback=epoch_callback)

    print("Testing stage")
    dataloader = DataLoader(ds_test, batch_size=1024, shuffle=False)
    autoencoder.eval()
    losses = 0.0
    for batch in dataloader:
        if cuda:
            batch = batch.cuda(non_blocking=True)
        output = autoencoder(batch)
        loss = F.mse_loss(output, batch)
        losses += loss.item() * batch.size(0)

    final_loss = losses / len(dataloader.sampler)
    print('final loss: ', final_loss)
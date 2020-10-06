import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import  os
from torch.autograd import Variable

from utils import *
from model import *


def parse_args():

    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    parser.add_argument('--dataroot', type=str, default="nasdaq/1.csv", help='path to dataset')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    parser.add_argument('--nhidden_encoder', type=int, default=32, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=32, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=24, help='the number of time steps in the window T [10, 24]')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train [100, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')
    parser.add_argument('--save_path', type = str, default="nasdaq", help='path to save predicted data')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    print("==> Load dataset ...")
    X, y = read_data(args.dataroot, debug=False)

    print("==> Initialize DA-RNN model ...")
    model = DA_RNN(
        X,
        y,
        args.ntimestep,
        args.nhidden_encoder,
        args.nhidden_decoder,
        args.batchsize,
        args.lr,
        args.epochs,
        args.save_path,
    )


    print("==> Start training ...")
    model.train()


    y_pred = model.test()
    y_pred += np.mean(model.orig_y[:model.train_timesteps])
    write_test_predict_data(args.save_path,y_pred)
    label = model.orig_y[model.train_timesteps:]
    rmse = np.sqrt(np.mean(np.subtract(y_pred, label) ** 2))
    mae = np.mean(np.abs(y_pred - label))
    mape = np.mean(np.abs(np.subtract(y_pred, label) / label))
    with open(os.path.join(args.save_path, 'index_test.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([rmse, mae, mape])
        csvfile.close()

    fig1 = plt.figure()
    plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    plt.savefig("1.png")
    plt.close(fig1)

    fig2 = plt.figure()
    plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    plt.savefig("2.png")
    plt.close(fig2)

    fig3 = plt.figure()
    plt.plot(y_pred, label='Predicted')
    plt.plot(model.orig_y[model.train_timesteps:], label="True")
    plt.legend(loc='upper left')
    plt.savefig("3.png")
    plt.close(fig3)
    print('Finished Training')

def write_test_predict_data(save_path,data):
    with open(os.path.join(save_path,'predict_data_test.csv'),'w',newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)
        csvfile.close()

if __name__ == '__main__':
    main()

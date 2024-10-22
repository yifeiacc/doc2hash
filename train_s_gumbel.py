import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from utils import *
from models.VDSH_S_gubel import VDSH_S_gubel
import argparse

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=100, type=int)
parser.add_argument("--lr", default=0.0001, type=float)
parser.add_argument('--single-label', dest='single_label', action='store_true',
                    help='Use softmax on the output of the prediction layer.')
parser.add_argument('--multi-label', dest='single_label', action='store_false',
                    help='Use sigmoid on the output of the prediction layer.')
parser.add_argument('--pred_weight', default=150.0, type=float, help='The weight of the prediction loss.')
parser.set_defaults(single_label=True)

args = parser.parse_args()

if not args.gpunum:
    parser.error("Need to provide the GPU number.")

if not args.dataset:
    parser.error("Need to provide the dataset.")

if not args.nbits:
    parser.error("Need to provide the number of bits.")

##################################################################################################

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpunum
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#########################################################################################################

dataset, data_fmt = args.dataset.split('.')

if dataset in ['reuters', 'tmc', 'rcv1']:
    if args.single_label:
        # parser.error("Must provide the multi-label flags since the dataset is multi-labeled.")
        print('set single_label flag to FALSE')
        args.single_label = False  # automatically set this flags

if args.single_label:
    train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)
else:
    train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt, download=True)
    test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt, download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

#########################################################################################################
y_dim = train_set.num_classes()
num_bits = args.nbits
num_features = train_set[0][0].size(0)

print("Train VDSH-S model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
if args.single_label:
    print("single-label prediction.")
else:
    print("multi-label prediction.")
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))
print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################
model = VDSH_S_gubel(dataset, num_features, num_bits, y_dim, device, dropoutProb=args.dropout, use_softmax=args.single_label)
model.to(device)

num_epochs = args.num_epochs

optimizer = optim.Adam(model.parameters(), lr=args.lr)
kl_weight = 2.
kl_step = 1 / 5000.
pred_weight = 0.
pred_weight_step = 3 / 1000.

tmp = 0.2
ar = 0.00003
print("tmp")


best_precision = 0
best_precision_epoch = 0

with open('logs/VDSH_S/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')

    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.to(device)
            yb = yb.to(device)

            logprob_w, qy, score_c = model(xb, tmp)
            kl_loss = VDSH_S_gubel.calculate_KL_loss(qy)
            reconstr_loss = VDSH_S_gubel.compute_reconstr_loss(logprob_w, xb)

            if args.single_label:
                pred_loss = model.compute_prediction_loss(score_c, yb)
            else:
                if len(yb.size()) == 1:
                    y_onehot = torch.zeros((xb.size(0), y_dim)).to(device)
                    y_onehot = y_onehot.scatter_(1, yb.unsqueeze(1), 1)
                    pred_loss = model.compute_prediction_loss(score_c, y_onehot)
                else:
                    pred_loss = model.compute_prediction_loss(score_c, yb)

            loss = reconstr_loss + kl_weight * kl_loss + pred_weight * pred_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tmp = max(tmp * 0.96, 0.1)
            # kl_weight = min(kl_weight + kl_step, 1.)
            pred_weight = min(pred_weight + pred_weight_step, 150.)
            avg_loss.append(loss.item())

            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(),
                                                                 reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.3f}'.format(model.get_name(), epoch + 1, np.mean(avg_loss),
                                                                         best_precision_epoch, best_precision))

        with torch.no_grad():
            train_b, test_b, train_y, test_y, plot_z = model.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100,
                                          is_single_label=args.single_label)
            print("precision at 100: {:.4f}".format(prec.item()))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1

#########################################################################################################
with open('logs/VDSH_S/result.txt', 'a') as handle:
    handle.write('{},{},{},{},{}\n'.format(dataset, data_fmt, args.nbits, best_precision_epoch, best_precision))

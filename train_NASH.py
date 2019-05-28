import os
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from datasets import *
from models.VDSH_gubel import VDSH_gubel
from utils import *
from models.VDSH import VDSH
import argparse
from models.NASH import *

##################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
parser.add_argument("-d", "--dataset", help="Name of the dataset.")
parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.2, type=float)
parser.add_argument("--train_batch_size", default=100, type=int)
parser.add_argument("--test_batch_size", default=100, type=int)
parser.add_argument("--transform_batch_size", default=100, type=int)
parser.add_argument("--num_epochs", default=20 , type=int)
parser.add_argument("--lr", default=0.001, type=float)

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
    single_label_flag = False
else:
    single_label_flag = True

if single_label_flag:
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

print("Train NASH model ...")
print("dataset: {}".format(args.dataset))
print("numbits: {}".format(args.nbits))
print("gpu id:  {}".format(args.gpunum))
print("dropout probability: {}".format(args.dropout))
if single_label_flag:
    print("single-label prediction.")
else:
    print("multi-label prediction.")
print("num epochs: {}".format(args.num_epochs))
print("learning rate: {}".format(args.lr))
print("num train: {} num test: {}".format(len(train_set), len(test_set)))

#########################################################################################################

Encoder = encoder(dataset, num_features, num_bits, dropoutProb=args.dropout, device=device)
Encoder.to(device)

Decoder = decoder(dataset, num_features, num_bits, dropoutProb=args.dropout, device=device)
Decoder.to(device)



num_epochs = args.num_epochs

optimizer_encoder = optim.Adam(Encoder.parameters(), lr=args.lr)
optimizer_decoder = optim.Adam(Decoder.parameters(), lr=args.lr)
kl_weight = 0.01
#kl_step = 1/5000

tmp = 1
ar = 0.00003
print("tmp")

best_precision = 0
best_precision_epoch = 0

with open('logs/VDSH/loss.log.txt', 'w') as log_handle:
    log_handle.write('epoch,step,loss,reconstr_loss,kl_loss\n')

    for epoch in range(num_epochs):
        avg_loss = []
        for step, (xb, yb) in enumerate(train_loader):
            Encoder.eval()
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            xb = xb.to(device)
            yb = yb.to(device)
            q = Encoder(xb, tmp)
            c = q.detach().requires_grad_()
            b = torch.sign(c-0.5).detach().requires_grad_()
            logprob_w = Decoder((b+1)/2)
            kl_loss = Decoder.calculate_KL_loss(c)
            reconstr_loss = Decoder.compute_reconstr_loss(logprob_w, xb)
            loss = reconstr_loss + kl_weight * kl_loss
            loss.backward()
            optimizer_decoder.step()
            Encoder.train()
            #print(b)
            q.backward(b.grad+c.grad)
            optimizer_encoder.step()
            #tmp = max(tmp*0.96, 0.1)
            #kl_weight = min(kl_weight*0.96, 1)




            avg_loss.append(loss.item())

            log_handle.write('{},{},{:.4f},{:.4f},{:.4f}'.format(epoch, step, loss.item(),
                                                                 reconstr_loss.item(), kl_loss.item()))
        print('{} epoch:{} loss:{:.4f} Best Precision:({}){:.3f}'.format(Encoder.get_name(), epoch + 1, np.mean(avg_loss),
                                                                         best_precision_epoch, best_precision))

        with torch.no_grad():
            train_b, test_b, train_y, test_y, plot_z = Encoder.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100,
                                          is_single_label=single_label_flag)
            print("precision at 100: {:.4f}".format(prec.item()))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1
                with open("plot_z_best.pk","wb") as f:
                    pickle.dump(plot_z, f)

            if epoch == num_epochs - 1:
                with open("plot_z_last.pk","wb") as f:
                    pickle.dump(plot_z, f)

#########################################################################################################
with open('logs/VDSH/result.txt', 'a') as handle:
    handle.write('{},{},{},{},{}\n'.format(dataset, data_fmt, args.nbits, best_precision_epoch, best_precision))
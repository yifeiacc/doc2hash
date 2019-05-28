import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class decoder(nn.Module):
    def __init__(self, dataset, vocabSize, latentDim, device, dropoutProb=0.):
        super(decoder, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device

        self.decoder = nn.Sequential(nn.Linear(self.latentDim, self.vocabSize),
                                     nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.decoder(x)
        return x

    def calculate_KL_loss(self,qy, categorical_dim=2):
        log_qy = torch.log(qy + 1e-20)
        g = Variable(torch.log(torch.Tensor([1.0 / categorical_dim]).cuda()))
        KLD = torch.sum(qy * (log_qy - g), dim=-1).mean()
        return KLD

    def compute_reconstr_loss(self, logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))


class encoder(nn.Module):

    def __init__(self, dataset, vocabSize, latentDim, device, dropoutProb=0.):
        super(encoder, self).__init__()

        self.dataset = dataset
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.dropoutProb = dropoutProb
        self.device = device

        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb),
                                     nn.Linear(self.hidden_dim, self.latentDim),
                                     nn.Sigmoid(),
                                     )

    def forward(self, document_mat, tmp):
        q = self.encoder(document_mat)
        return q

    def get_name(self):
        return "VDSH"

    def get_binary_code(self, train, test):
        train_zy = []
        for xb, yb in train:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim)
            b = (torch.sign(q_y - 0.5) + 1) / 2
            train_zy.append((b, yb))
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0).type(torch.cuda.ByteTensor)
        train_y = torch.cat(train_y, dim=0)

        test_zy = []
        tmp_z = []
        for xb, yb in test:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim)
            b = (torch.sign(q_y - 0.5) + 1) / 2
            test_zy.append((b, yb))
            tmp_z.append(q)
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0).type(torch.cuda.ByteTensor)
        test_y = torch.cat(test_y, dim=0)
        plot_z = torch.cat(tmp_z, dim=0)

        # mid_val, _ = torch.median(train_z, dim=0)
        # train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        # test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        # del train_z
        # del test_z
        return train_z, test_z, train_y, test_y, plot_z
        # return train_b, test_b, train_y, test_y

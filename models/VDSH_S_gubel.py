import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()
    return -Variable(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, latent_dim, categorical_dim = 2):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1,latent_dim*categorical_dim)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class VDSH_S_gubel(nn.Module):

    def __init__(self, dataset, vocabSize, latentDim, num_classes, device, dropoutProb=0., use_softmax=True):
        super(VDSH_S_gubel, self).__init__()
        self.dataset = dataset
        self.hidden_dim = 500
        self.vocabSize = vocabSize
        self.latentDim = latentDim
        self.num_classes = num_classes
        self.dropoutProb = dropoutProb
        self.device = device
        self.use_softmax = use_softmax




        self.encoder = nn.Sequential(nn.Linear(self.vocabSize, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(p=dropoutProb),
                                     nn.Linear(self.hidden_dim, self.latentDim * 2),
                                     )


        self.decoder = nn.Sequential(nn.Linear(self.latentDim*2, self.vocabSize),
                                     nn.LogSoftmax(dim=1))


        if use_softmax:
            self.pred = nn.Sequential(nn.Linear(self.latentDim*2, self.num_classes))
            self.pred_loss = nn.CrossEntropyLoss() # combine log_softmax and NLLloss
        else:
            self.pred = nn.Sequential(nn.Linear(self.latentDim*2, self.num_classes),
                                      nn.Sigmoid())

        self.encoder.apply(init_weights)

        self.decoder.apply(init_weights)
        self.pred.apply(init_weights)

    def gumbel_softmax(logits, temperature, latent_dim, categorical_dim=2):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        y = gumbel_softmax_sample(logits, temperature)
        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        y_hard = (y_hard - y).detach() + y
        return y_hard.view(-1, latent_dim * categorical_dim)

    def forward(self, document_mat, tmp):
        q = self.encoder(document_mat)
        q_y = q.view(q.size(0), self.latentDim ,2)
        z = gumbel_softmax(q_y, tmp, latent_dim=self.latentDim)
        prob_w = self.decoder(z)
        score_c = self.pred(z)
        return prob_w, F.softmax(q), score_c

    def get_name(self):
        return "VDSH_S_gumbel"

    @staticmethod
    def calculate_KL_loss(qy, categorical_dim=2):
        log_qy = torch.log(qy+1e-20)
        g = Variable(torch.log(torch.Tensor([1.0/categorical_dim]).cuda()))
        KLD = torch.sum(qy*(log_qy-g),dim=-1).mean()
        return KLD

    @staticmethod
    def compute_reconstr_loss(logprob_word, doc_mat):
        return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))

    def compute_prediction_loss(self, scores, labels):
        if self.use_softmax:
            return self.pred_loss(scores, labels)
        else:
            # compute L2 distance
            return torch.mean(torch.sum((scores - labels) ** 2., dim=1))

    def get_binary_code(self, train, test):
        train_zy = []
        for xb, yb in train:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y,dim=2)
            train_zy.append((b,yb))
        train_z, train_y = zip(*train_zy)
        train_z = torch.cat(train_z, dim=0)
        train_y = torch.cat(train_y, dim=0)

        test_zy = []
        tmp_z = []
        for xb, yb in test:
            q = self.encoder(xb.to(self.device))
            q_y = q.view(q.size(0), self.latentDim, 2)
            b = torch.argmax(q_y, dim=2)
            test_zy.append((b, yb))
            tmp_z.append(q)
        test_z, test_y = zip(*test_zy)
        test_z = torch.cat(test_z, dim=0)
        test_y = torch.cat(test_y, dim=0)
        plot_z = torch.cat(tmp_z, dim = 0)

        # mid_val, _ = torch.median(train_z, dim=0)
        # train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        # test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        # del train_z
        # del test_z
        return train_z, test_z, train_y, test_y, plot_z
        #return train_b, test_b, train_y, test_y
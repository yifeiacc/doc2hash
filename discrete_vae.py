import math

import torch
import torch.nn as nn

from DVAE import datas
from DVAE.utils import to_var, ListModule, print_results, kl_multinomial
import torch.nn.functional as F

import torch
import torch.optim as optim
from datasets import *
from models.VDSH_gubel import VDSH_gubel
from utils import *
import argparse

class Encoder(nn.Module):
    def __init__(self, image_size=784, N=6, K=6, M=20):
        super(Encoder, self).__init__()
        self.hidden_dim = 300
        self.N = N
        self.K = K
        self.M = M
        self.encoder = nn.Sequential(nn.Linear(image_size, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(inplace=True),
                                     # nn.Dropout(p=dropoutProb),
                                     nn.Linear(self.hidden_dim, N * K),
                                     )
        # self.encoder = nn.Sequential(
        #     nn.Linear(image_size, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, N * K))

    def forward(self, x):
        phi_x = self.encoder(x).view(-1, self.K)
        z, phi_x_g, hascode = self.gumbel_perturbation(phi_x)
        return z, phi_x_g, phi_x, hascode

    def sample_gumbel(self, shape, eps=1e-20):
        # Sample from Gumbel(0, 1)
        U = torch.rand(shape).float()
        return -torch.log(eps - torch.log(U + eps))

    def gumbel_perturbation(self, phi_x, eps=1e-10):
        M, K, N = self.M, self.K, self.N

        phi_x = phi_x.repeat(M, 1)
        shape = phi_x.size()
        gumbel_noise = to_var(self.sample_gumbel(shape, eps=eps))
        phi_x_gamma = phi_x + gumbel_noise
        # hard:
        _, k = phi_x_gamma.data.max(-1)
        hashcode = k.view(-1, self.N)

        z_phi_gamma = to_var(torch.FloatTensor(*shape)).zero_().scatter_(-1, k.view(-1, 1), 1.0)


        return z_phi_gamma, phi_x_gamma, hashcode


class Decoder(nn.Module):
    def __init__(self, image_size=784, N=6, K=6, composed_decoder=True):
        super(Decoder, self).__init__()
        self.composed_decoder = composed_decoder

        if not composed_decoder:
            linear_combines = [nn.Sequential(nn.Linear(K, 300),
                                             nn.ReLU(),
                                             nn.Linear(300, image_size)) for _ in range(N)]
            self.decoder = ListModule(*linear_combines)
        else:
            self.decoder = nn.Sequential(
                nn.Linear(N * K, image_size),
                nn.LogSoftmax(dim=1),
                # nn.ReLU(),
                # nn.Linear(300, image_size)
            )

        self.N = N
        self.K = K

    def forward(self, y):

        if not self.composed_decoder:
            y = y.view(-1, self.N, self.K)
            bs = y.size(0)
            n_out = []
            for n in range(self.N):
                nth_input = y[:, n, :].view(-1, self.K)
                n_out.append(F.sigmoid(self.decoder[n](nth_input)))

            out = torch.stack(n_out, 1).mean(1)
        else:
            out = self.decoder(y.view(-1, self.N * self.K))
        return out


class Direct_VAE:
    def __init__(self, params):

        self.N, self.K = params['N_K']
        self.M = params['gumbels']
        self.encoder = Encoder(image_size= params['input_size'], N=self.N, K=self.K, M=self.M).to(params["device"])
        self.decoder = Decoder(image_size= params['input_size'], N=self.N, K=self.K, composed_decoder=params['composed_decoder']).to(params["device"])
        self.eps = params['eps_0']
        self.annealing_rate = params['anneal_rate']

        self.params = params

        print('encoder: ', self.encoder)
        print('decoder: ', self.decoder)

        if torch.cuda.is_available():
            self.encoder.cuda()
            self.decoder.cuda()

        lr = params['learning_rate']
        self.optimizer_e = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.optimizer_d = torch.optim.Adam(self.decoder.parameters(), lr=lr)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.training_iter = 0

    def compute_reconstr_loss(self, logprob_word, doc_mat):
            return -torch.mean(torch.sum(logprob_word * doc_mat, dim=1))
    def train(self, train_loader):
        kl_sum, bce_sum = 0, 0
        eps_0, ANNEAL_RATE, min_eps = self.params['eps_0'], self.params['anneal_rate'], self.params['min_eps']
        for i, (im, _) in enumerate(train_loader):
            images = im.view(im.size(0), -1).to(self.params["device"])
            bs = im.size(0)
            ground_truth = images.repeat(self.M, 1)
            # forward
            z_hard, phi_x_g, phi_x,_ = self.encoder(images)
            out = self.decoder(z_hard)

            # backward
            gradients = self.compute_encoder_gradients(z_hard, phi_x_g, ground_truth, self.eps)
            encoder_loss = torch.sum(to_var(gradients) * phi_x_g)

            #decoder_loss = self.bce_loss(out, ground_truth).view(self.M, bs, -1).mean(0).sum()
            decoder_loss = self.compute_reconstr_loss(out, ground_truth)
            kl = kl_multinomial(phi_x)

            # decoder_loss += kl
            self.optimizer_e.zero_grad()
            self.optimizer_d.zero_grad()

            encoder_loss.backward()
            decoder_loss.backward()

            self.optimizer_d.step()
            self.optimizer_e.step()

            bce_sum += (decoder_loss + kl).detach() / bs

            if self.training_iter % 500 == 0:
                a = eps_0 * math.exp(-ANNEAL_RATE * self.training_iter)
                self.eps = np.maximum(a, min_eps).item()
            self.training_iter += 1

        nll_bce = (bce_sum.item()) / len(train_loader)
        return nll_bce

    def evaluate(self, test_loader):
        self.encoder.eval()
        self.decoder.eval()
        # self.encoder.M = 100
        # self.decoder.M = 100
        # self.M = 100
        bce_sum = 0
        kl_div = 0
        ret = []
        with torch.no_grad():
            for images, yb in test_loader:
                images = images.view(images.size(0), -1).to(self.params["device"])
                yb = yb.to(self.params["device"])
                ground_truth = images.repeat(self.M, 1)
                bs = images.size(0)
                hards, _, phi_x, hashcode = self.encoder(images)
                out = self.decoder(hards)

                decoder_loss = self.bce_loss(out, ground_truth).view(self.M, bs, -1).mean(0).sum()
                kl = kl_multinomial(phi_x)
                bce_sum += (decoder_loss + kl) / images.size(0)
                ret.append((hashcode, yb))
        data_z,data_y = zip(*ret)
        z = torch.cat(data_z, dim=0)
        yz = torch.cat(data_y, dim=0)


        self.encoder.train()
        self.decoder.train()
        self.encoder.M = self.params['gumbels']
        self.decoder.M = self.params['gumbels']
        self.M = self.params['gumbels']
        nll_bce = bce_sum.item() / len(test_loader)
        return nll_bce, z, yz

    def compute_encoder_gradients(self, z_hard, phi_x_g, ground_truth, epsilon=1.0):
        with torch.no_grad():
            N = self.N
            K = self.K
            soft_copy = phi_x_g.data
            hard_copy = z_hard.data.view(-1, N, K)
            self.decoder.eval()

            new_batch = []
            gt_batch = []
            for n in range(N):
                a_clone = hard_copy.clone()
                idx = to_var(n * torch.ones(hard_copy.size(0), 1, hard_copy.size(2)).long())
                a_clone.scatter_(1, idx, 0)

                for k in range(K):
                    clone2 = a_clone.clone()
                    clone2[:, n, k] = 1
                    new_batch.append(clone2)
                    gt_batch.append(ground_truth)

            new_batch = torch.cat(new_batch, 1)
            gt_batch = torch.cat(gt_batch, 1).view(-1, ground_truth.size(-1))

            out = self.decoder(to_var(new_batch))
            losses = self.bce_loss(out, gt_batch).sum(dim=1)  # ground_truth.repeat(K*N,1)
            hard_copy = hard_copy.view(-1, K)
            losses = epsilon * losses.view(-1, K).data
            soft_copy = soft_copy - losses
            shape = soft_copy.size()
            _, k = soft_copy.max(-1)

            change = to_var(torch.FloatTensor(*shape).zero_()).scatter_(-1, k.view(-1, 1), 1.0)
            gradients = hard_copy - change
            self.decoder.train()
            gradients = gradients * (1.0 / epsilon)
        return gradients

    def get_binary_code(self, train, test):
        # train_zy = []
        # for xb, yb in train:
        #     q = self.encoder(xb.to(self.device))
        #     q_y = q.view(q.size(0), self.latentDim, 2)
        #     b = torch.argmax(q_y,dim=2)
        #     train_zy.append((b,yb))
        # train_z, train_y = zip(*train_zy)
        # train_z = torch.cat(train_z, dim=0)
        # train_y = torch.cat(train_y, dim=0)
        #
        # test_zy = []
        # tmp_z = []
        # for xb, yb in test:
        #     q = self.encoder(xb.to(self.device))
        #     q_y = q.view(q.size(0), self.latentDim, 2)
        #     b = torch.argmax(q_y, dim=2)
        #     test_zy.append((b, yb))
        #     tmp_z.append(q)
        # test_z, test_y = zip(*test_zy)
        # test_z = torch.cat(test_z, dim=0)
        # test_y = torch.cat(test_y, dim=0)
        # plot_z = torch.cat(tmp_z, dim = 0)

        # mid_val, _ = torch.median(train_z, dim=0)
        # train_b = (train_z > mid_val).type(torch.cuda.ByteTensor)
        # test_b = (test_z > mid_val).type(torch.cuda.ByteTensor)

        # del train_z
        # del test_z

        _, train_z, train_y = self.evaluate(train)
        _, test_z, test_y = self.evaluate(test)

        return train_z, test_z, train_y, test_y, None


def training_procedure(params):
    """trains over the MNIST standard spilt (50K/10K/10K) or omniglot
    saves the best model on validation set
    evaluates over test set every epoch just for plots"""

    torch.manual_seed(params['random_seed'])

    train_loader = params['train_loader']
    valid_loader = params['valid_loader']
    test_loader = params['test_loader']

    N, K = params['N_K']
    direct_vae = Direct_VAE(params)

    best_state_dicts = None
    print('hyper parameters: ', params)

    train_results, valid_results, test_results = [], [], []
    best_valid, best_test_nll = float('Inf'), float('Inf')
    best_precision = 0
    best_precision_epoch = 0
    for epoch in range(params['num_epochs']):
        epoch_results = [0, 0, 0]
        train_nll = direct_vae.train(train_loader)
        train_results.append(train_nll)
        epoch_results[0] = train_nll

        valid_nll, _, _ = direct_vae.evaluate(valid_loader)
        valid_results.append(valid_nll)
        epoch_results[1] = valid_nll

        test_nll, _, _ = direct_vae.evaluate(test_loader)
        test_results.append(test_nll)
        epoch_results[2] = test_nll

        if params['print_result']:
            print_results(epoch_results, epoch, params['num_epochs'])

        if valid_nll < best_valid:
            best_valid = valid_nll
            best_test_nll = test_nll
            best_state_dicts = (direct_vae.encoder.state_dict(), direct_vae.decoder.state_dict())

        with torch.no_grad():
            train_b, test_b, train_y, test_y, plot_z = direct_vae.get_binary_code(train_loader, test_loader)
            retrieved_indices = retrieve_topk(test_b.to(device), train_b.to(device), topK=100)
            prec = compute_precision_at_k(retrieved_indices, test_y.to(device), train_y.to(device), topK=100,
                                          is_single_label=single_label_flag)
            print("precision at 100: {:.4f}".format(prec.item()))

            if prec.item() > best_precision:
                best_precision = prec.item()
                best_precision_epoch = epoch + 1

    return train_results, test_results, best_test_nll, best_state_dicts, params.copy()



if __name__ == '__main__':
    import torch
    import torch.optim as optim
    from datasets import *
    from models.VDSH_gubel import VDSH_gubel
    from utils import *
    import argparse

    ##################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpunum", help="GPU number to train the model.")
    parser.add_argument("-d", "--dataset", help="Name of the dataset.")
    parser.add_argument("-b", "--nbits", help="Number of bits of the embedded vector.", type=int)
    parser.add_argument("--dropout", help="Dropout probability (0 means no dropout)", default=0.2, type=float)
    parser.add_argument("--train_batch_size", default=100, type=int)
    parser.add_argument("--test_batch_size", default=100, type=int)
    parser.add_argument("--transform_batch_size", default=100, type=int)
    parser.add_argument("--num_epochs", default=100, type=int)
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
        train_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt,
                                           download=True)
        test_set = SingleLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt,
                                          download=True)
    else:
        train_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='train', bow_format=data_fmt,
                                          download=True)
        test_set = MultiLabelTextDataset('dataset/{}'.format(dataset), subset='test', bow_format=data_fmt,
                                         download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=True)

    #########################################################################################################
    y_dim = train_set.num_classes()
    num_bits = args.nbits
    num_features = train_set[0][0].size(0)

    print("Train DVEA model ...")
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

    params = {
              'device': device,
              'input_size': num_features,
              'num_epochs':  args.num_epochs, # 300
              'composed_decoder': True,
              'batch_size': 100,
              'learning_rate': args.lr, #0.001,
              'gumbels': 1,
              'N_K': (num_bits, 2),
              'eps_0': 1.0,
              'anneal_rate': 1e-5, #1e-5,
              'min_eps': 0.1,
              'random_seed': 777,
              'dataset': 'mnist',  # 'mnist' or 'omniglot'
              'split_valid': True,
              'binarize': True,
              'ST-estimator': False,  # relevant only for GSM
              'save_images': False,
              'print_result': True,
              'train_loader': train_loader,
              'test_loader': test_loader,
              'valid_loader': test_loader,
              'drop_out_prob': args.dropout
    }

    dvae_results = training_procedure(params)
    dvae_results = training_procedure(params)
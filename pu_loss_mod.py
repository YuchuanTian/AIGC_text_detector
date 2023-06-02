import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable
from prior_kit import expectation_matrix

        
    
class PULossauto:
    def __init__(self):
        self.prior = 0
        self.label = 0

    # @staticmethod
    def apply(self, input, label, prior):
        self.input = input
        self.label = label
        if type(prior)==float: # self added
            prior = torch.tensor(prior)
        self.prior = prior.to(input.device).float()
        self.positive = 1
        self.unlabeled = -1
        self.loss_func = lambda x: F.sigmoid(-x) # this x is merely a real number
        self.beta = 0
        self.gamma = 1
        
        self.positive_x = (self.label==self.positive).float()
        self.unlabeled_x = (self.label==self.unlabeled).float()
        self.positive_num = torch.max(torch.sum(self.positive_x), torch.tensor(1).to(input.device).float())
        self.unlabeled_num = torch.max(torch.sum(self.unlabeled_x), torch.tensor(1).to(input.device).float())
        self.positive_y = self.loss_func(self.input)
        self.unlabeled_y = self.loss_func(-self.input) # all regarded as negative
        self.positive_loss = torch.sum(self.prior * self.positive_x / self.positive_num * self.positive_y.squeeze())
        self.negative_loss = torch.sum((self.unlabeled_x / self.unlabeled_num - self.prior * self.positive_x / self.positive_num) * self.unlabeled_y.squeeze())
        objective = self.positive_loss + self.negative_loss
        
        if self.negative_loss.data < -self.beta:
            objective = self.positive_loss - self.beta
            self.x_out = -self.gamma * self.negative_loss
        else:
            self.x_out = objective
        return objective


class pu_loss_auto():
    def __init__(self, prior, pu_type='', max_length=512, device='cpu'):
        self.prior = prior
        # self.label = label
        self.pu_type = pu_type
        self.device = device
        if pu_type in ['dual_softmax_dyn_dtrun']:
            self.loss_mod = PULossauto()
        else:
            raise NotImplementedError(f'PU type {pu_type} not implemented...')
        # for random walk:
        if pu_type in ['dual_softmax_dyn_dtrun']:
            expectations = list()
            for i in range(0, max_length+1):
                expectations.append(expectation_matrix(i, self.prior, device))
            self.prior = torch.stack(expectations)
            print('All dynamic priors calculated...')


    def __call__(self, input, label, sentence_length):
        prior = self.prior
        if 'dyn' in self.pu_type:
            prior = self.prior[sentence_length]
        return self.loss_mod.apply(input, label, prior)
    
    def logits_to_scores(self, logits):
        if self.pu_type in ['dual_softmax_dyn_dtrun']:
            return F.softmax(logits, dim=-1)[..., 0] # take human as positive
        else:
            raise NotImplementedError(f'PU type {self.pu_type} not implemented')
import numpy as np
import torch


def expectation_matrix(length, pi, device='cpu'):
    if length < 3:
        return torch.tensor(pi).float().to(device)
    state = torch.zeros((1, length+1)).float().to(device)
    state[0, 0] += 1.
    trans = torch.zeros((length+1,length+1)).float().to(device) # state transition matrix
    trans[1:, :-1] += torch.eye(length).to(device)*pi
    trans[:-1, 1:] += torch.eye(length).to(device)*(1-pi)
    trans[0,0] += pi
    trans[length, length] += (1-pi)

    total_trans = torch.zeros_like(trans) + torch.eye(length+1).to(device) # id mat
    for _ in range(length):
        total_trans @= trans
    distribution = (state @ total_trans).squeeze(0)
    expectation = 1. - ((distribution * torch.arange(0, length+1).to(device)).sum()/length)
    return expectation.to(device)
    

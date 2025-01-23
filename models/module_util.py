"""
Replications of module utils from Wortsman et al. SupSup
"""
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd

import math


def mask_init(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def pspinit(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores


def bn_mask_init(module):
    return torch.ones(module.num_features)


def bn_mask_initv2(module):
    return torch.zeros(module.num_features)


def rank_one_init(module):
    scores = torch.Tensor(module.weight.size(0))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def rank_one_initv2(module):
    scores = torch.Tensor(module.weight.size(1))
    nn.init.uniform_(scores, a=-1, b=1)
    scores = scores.sign().float()
    return scores


def mask_initv2(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_uniform_(scores, a=math.sqrt(5))
    return scores[0]


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # original_scores = scores.clone().flatten().cpu().numpy()

        # print("原始分数分布:")
        # print(np.histogram(original_scores, bins=10))

        # out = scores.clone()
        # _, idx = scores.flatten().sort()
        # j = int((1 - k) * scores.numel())
        #
        # threshold_score = original_scores[idx[j].item()]
        # print(f"阈值分数: {threshold_score}")
        #
        # selected_scores = original_scores[idx[j:].cpu().numpy()]
        # print("被选中的分数:")
        # print(np.histogram(selected_scores, bins=10))

        # 检查边界情况
        # scores_near_threshold = original_scores[
        #     (original_scores >= threshold_score * 0.99) & (original_scores <= threshold_score * 1.01)]
        # print(f"接近阈值的分数比例: {len(scores_near_threshold)/len(original_scores)}")

        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class GetSignedSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        ctx.save_for_backward(scores)
        out = scores.clone()
        _, idx = scores.abs().flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out * scores.sign()

    @staticmethod
    def backward(ctx, g):
        scores, = ctx.saved_tensors

        # send the gradient g straight-through on the backward pass.
        return g , None


def get_subnet(scores, k):
    out = scores.clone()
    _, idx = scores.flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out

def get_subnet_signed(scores, k):
    out = scores.clone()
    _, idx = scores.abs().flatten().sort()
    j = int((1 - k) * scores.numel())

    # flat_out and out access the same memory.
    flat_out = out.flatten()
    flat_out[idx[:j]] = 0
    flat_out[idx[j:]] = 1

    return out * scores.sign()


class GetSubnetFast(autograd.Function):
    @staticmethod
    def forward(ctx, scores, a=0):
        return (scores >= a).float()

    @staticmethod
    def backward(ctx, g):
        return g, None


def get_subnet_fast(scores, a=0):
    return (scores >= a).float()


def kaiming_normal(module):
    scores = torch.Tensor(module.weight.size())
    nn.init.kaiming_normal_(scores, nonlinearity="relu")
    return scores
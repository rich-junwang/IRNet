#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import numpy as np


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def build_future_mask(in_tensor, time_dim=0):
    seq_len = in_tensor.size(time_dim)
    future_mask = torch.zeros(seq_len, seq_len, requires_grad=False, device=in_tensor.device)
    future_mask = torch.triu(fill_with_neg_inf(future_mask), 1)
    return future_mask


def make_positions(tensor, padding_idx, offset=0):
    """Replace non-padding symbols with their position numbers.
    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    "<s>" : 0,
    "<pad>" : 1,
    "</s>" : 2,
    "<unk>" : 3,
    "." : 4,
    "Ġthe" : 5,
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    positions = torch.cumsum(mask, dim=1).type_as(mask) + offset
    return (positions * mask).long() + padding_idx  # Position numbers begin at padding_idx+1.


def compute_loss(logits, targets, mask, reduce=True):
    '''
    :param logits: [batch_size, seq_len, vocab_size]
    :param targets: [batch_size, seq_len]
    :param mask: [batch_size, seq_len]
    :return:
    '''
    # print(logits.size())
    # print(targets.size())
    from torch.nn import CrossEntropyLoss
    ce_loss_layer = CrossEntropyLoss(reduction='none')
    batch_size, seq_len, vocab_size = logits.size()
    logits = logits.view(batch_size * seq_len, vocab_size)
    targets = targets.contiguous().view(batch_size * seq_len)
    loss = ce_loss_layer(logits, targets)  # [batch_size * seq_len]
    loss = loss.view(batch_size, seq_len) * mask
    if reduce:
        loss = loss.sum(dim=-1).mean(dim=-1)
    return loss


def compute_loss2(logits, targets, mask):
    '''
    :param logits: [batch_size, seq_len, vocab_size]
    :param targets: [batch_size, seq_len]
    :param mask: [batch_size, seq_len]
    :return:
    '''
    batch_size, seq_len, vocab_size = logits.size()
    lprobs = F.log_softmax(logits.float(), dim=-1)
    lprobs = lprobs.view(-1, lprobs.size(-1))
    loss = F.nll_loss(
        lprobs,
        targets.view(-1),
        reduction='none',
    )
    loss = loss.view(batch_size, seq_len) * mask
    loss = loss.sum(dim=-1).mean(dim=-1)
    return loss


def compute_loss_bak(logits, targets, epsilon=0.1, ignore_index=None, reduce=True):
    log_probs = F.log_softmax(logits.float(), dim=-1)
    log_probs = log_probs.view(-1, log_probs.size(-1))
    if targets.dim() == log_probs.dim() - 1:
        targets = targets.unsqueeze(-1)
    nll_loss = -log_probs.gather(dim=-1, index=targets)
    smooth_loss = -log_probs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = targets.eq(ignore_index)
        if pad_mask.any():
            nll_loss.masked_fill_(pad_mask, 0.)
            smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / log_probs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    sample_size = targets.size(0)
    return loss / sample_size


def check_correctness(predictions, truths, mask):
    correctness = (predictions == truths) & mask
    total = int(mask.sum().data.cpu().numpy())
    correct = int(correctness.sum().data.cpu().numpy())
    return total, correct


def sequence_loss_mask(predictions, truths, mask=None):
    correctness = predictions == truths
    acc_correctness = torch.cumsum(correctness, dim=1)
    positions = torch.ones(correctness.size(), dtype=torch.int, device=correctness.device, requires_grad=False)
    positions = torch.cumsum(positions, dim=1) - 1  # shift left
    # print('acc_correctness:', acc_correctness)
    # print('positions:', positions)
    # seq_mask = acc_correctness >= positions
    seq_mask = acc_correctness == positions
    if mask is not None:
        seq_mask &= mask
    return seq_mask


def sequence_correctness(predictions, truths, mask=None):
    correctness = predictions == truths
    acc_correctness = torch.cumsum(correctness, dim=1)
    positions = torch.ones(correctness.size(), dtype=torch.int, device=correctness.device, requires_grad=False)
    positions = torch.cumsum(positions, dim=1) - 1  # shift left
    seq_mask = acc_correctness >= positions
    if mask is not None:
        seq_mask &= mask
    correct = int(seq_mask.sum().data.cpu().numpy())
    return correct


def compute_seq_loss(
        seq_log_probs,  # [batch_size, action_size]
        valid_positions_mask,  # [batch_size, action_size]
        neg_seq_log_probs,   # [batch_size, action_size]
        neg_valid_positions_mask, # [batch_size, action_size]
        margin=1.0,
    ):
    seq_log_probs = seq_log_probs * valid_positions_mask
    cum_log_probs = torch.cumsum(seq_log_probs, dim=-1)
    neg_seq_log_probs = neg_seq_log_probs * neg_valid_positions_mask
    neg_cum_log_probs = torch.cumsum(neg_seq_log_probs, dim=-1)
    mask = valid_positions_mask | neg_valid_positions_mask

    # positions = torch.ones(seq_log_probs.size(), dtype=torch.int, device=seq_log_probs.device, requires_grad=False)
    # positions = torch.cumsum(positions, dim=1)
    # loss = torch.clamp(positions * margin + neg_cum_log_probs - cum_log_probs, min=0.0) * mask

    loss = torch.clamp(margin + neg_cum_log_probs - cum_log_probs, min=0.0) * mask

    # print("seq_log_probs:", seq_log_probs)
    # print("valid_positions_mask:", valid_positions_mask)
    # print("neg_seq_log_probs:", neg_seq_log_probs)
    # print("neg_valid_positions_mask:", neg_valid_positions_mask)
    # print("cum_log_probs:", cum_log_probs)
    # print("neg_cum_log_probs:", neg_cum_log_probs)
    # print("mask:", mask)
    # print("loss:", loss)
    # loss = loss.sum(dim=-1).mean(dim=-1)
    loss = loss.mean()
    return loss


if __name__ == '__main__':
    # a = torch.zeros(6, 4)
    # future_mask = build_future_mask(a, time_dim=1)
    # print(future_mask)

    '''
    a = [
        [5, 4, 3, 2, 1, 0, 0],
        [5, 4, 3, 2, 0, 0, 0],
        [5, 4, 3, 0, 0, 0, 0],
    ]
    a = np.array(a, dtype=np.float32)
    a_tensor = torch.from_numpy(a)
    print(a_tensor)
    positions = make_positions(a_tensor, 0, offset=5)
    print(positions)
    '''

    '''
    prediction = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5],
    ]
    prediction = torch.from_numpy(np.array(prediction, dtype=np.int32))

    truth = [
        [1, 2, 3, 4, 5],
        [1, 2, 3, 3, 5],
    ]
    truth = torch.from_numpy(np.array(truth, dtype=np.int32))

    padding_mask = [
        [True, True, True, True, True],
        [True, True, True, True, False],
    ]
    padding_mask = torch.from_numpy(np.array(padding_mask, dtype=bool))
    print(check_correctness(prediction, truth, padding_mask))
    '''

    # compute mask based on prediction and ground-truth
    # logits: [batch_size, seq_len, vocab_size]
    # truth: [batch_size, seq_len]

    '''
    batch_size = 3
    seq_len = 5
    vocab_len = 10
    logits = np.random.rand(batch_size, seq_len, vocab_len)
    logits = torch.from_numpy(np.array(logits, dtype=np.float32))
    truth = np.random.randint(vocab_len, size=(batch_size,seq_len))
    truth = torch.from_numpy(np.array(truth, dtype=np.int32))

    predictions = torch.argmax(logits, dim=-1)

    correctness = (predictions == truth)
    print("truth:", truth)
    print("prediction:", predictions)
    print("correctness", correctness)
    result = sequence_loss_mask(predictions, truth)
    print(result)
    #'''
    '''
    correctness = [
    [True, True, False, False, False],
    [True, True, True, False, False],
    [True, True, True, False, False]]
    correctness = torch.from_numpy(np.array(correctness, dtype=np.bool))
    acc_correctness = torch.cumsum(correctness, dim=1)
    positions = torch.ones(correctness.size(), dtype=torch.int, device=correctness.device, requires_grad=False)
    positions = torch.cumsum(positions, dim=1) - 1  # shift left
    mask = acc_correctness == positions
    print("correctness", correctness)
    print("acc_correctness:", acc_correctness)
    print("positions:", positions)
    print("mask:", mask)
    # '''

    batch_size = 3
    action_size = 5
    seq_log_probs = torch.from_numpy(np.array(np.random.rand(batch_size, action_size), dtype=np.float32))
    neg_seq_log_probs = torch.from_numpy(np.array(np.random.rand(batch_size, action_size), dtype=np.float32))

    valid_positions_mask = [
        [True, True, False, False, False],
        [True, True, True, False, False],
        [True, True, True, False, False]]
    valid_positions_mask = torch.from_numpy(np.array(valid_positions_mask, dtype=np.bool))

    neg_valid_positions_mask = [
        [True, True, True, False, False],
        [True, True, True, True, False],
        [True, True, True, True, True]]
    neg_valid_positions_mask = torch.from_numpy(np.array(neg_valid_positions_mask, dtype=np.bool))

    loss = compute_seq_loss(
        seq_log_probs,  # [batch_size, action_size]
        valid_positions_mask,  # [batch_size, action_size]
        neg_seq_log_probs,  # [batch_size, action_size]
        neg_valid_positions_mask)



    print('DONE!')

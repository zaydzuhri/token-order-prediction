import torch
import torch.nn.functional as F

# Naive ListNet loss function implementation
def list_net_loss(y_pred, y_true):
    """
    ListNet loss introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".
    :param y_pred: predictions from the model, shape [*, slate_length]
    :param y_true: ground truth labels, shape [*, slate_length]
    :return: loss value, a torch.Tensor
    """
    return torch.mean(-torch.sum(F.softmax(y_true, dim=-1).nan_to_num(nan=0) * F.log_softmax(y_pred, dim=-1), dim=-1))
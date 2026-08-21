import torch
import torch.nn.functional as F
import torch
import torch.nn.functional as F
from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss


def mobile_bce_dice_loss(predictions, target):
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    loss, eps = 0.0, 1e-5
    for prediction in predictions:
        probabilities = torch.softmax(prediction, dim=1)[:, 1:2]
        bce = F.binary_cross_entropy(probabilities, target)
        intersection = (probabilities * target).sum()
        dice = (2 * intersection + eps) / (probabilities.sum() + target.sum() + eps)
        loss = loss + bce + 1 - dice
    return loss

def mobile_bce_dice_loss(predictions, target):
    if not isinstance(predictions,(list,tuple)): predictions=[predictions]
    if target.ndim==3: target=target.unsqueeze(1)
    target=target.float(); total=0.0; eps=1e-5
    for prediction in predictions:
        p=torch.softmax(prediction,dim=1)[:,1:2]
        total += F.binary_cross_entropy(p,target) + 1-(2*(p*target).sum()+eps)/(p.sum()+target.sum()+eps)
    return total

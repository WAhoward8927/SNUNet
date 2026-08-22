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
    # Mobile-CDNet has one probability output.  SNUNet's final decoder output
    # is therefore the only tensor used for the comparable objective.
    prediction = predictions[-1] if isinstance(predictions, (list, tuple)) else predictions
    if target.ndim == 3:
        target = target.unsqueeze(1)
    target = target.float()
    probability = torch.softmax(prediction, dim=1)[:, 1:2]
    bce = F.binary_cross_entropy(probability, target)
    eps = 1e-5
    dice = (2 * (probability * target).sum() + eps) / (probability.sum() + target.sum() + eps)
    return bce + 1 - dice

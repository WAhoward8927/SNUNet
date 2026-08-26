from utils.metrics import FocalLoss, dice_loss

def hybrid_loss(predictions, target):
    prediction = predictions[-1]
    return FocalLoss(gamma=0, alpha=None)(prediction, target) + dice_loss(prediction, target)

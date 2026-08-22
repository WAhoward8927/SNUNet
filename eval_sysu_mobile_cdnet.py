import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = '/content/drive/MyDrive/SNUNet/SYSU/checkpoints/checkpoint_epoch_199.pt'
model = torch.load(path, weights_only=False)

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten(), labels=[0,1]).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))

from pathlib import Path
import json, numpy as np, scipy.io as scio
report=Path('/content/drive/MyDrive/SNUNet/SYSU/reports'); (report/"Vis").mkdir(parents=True,exist_ok=True)
tn,fp,fn,tp=[int(c_matrix[k]) for k in ("tn","fp","fn","tp")]
eps=np.finfo(np.float32).eps
recall=tp/(tp+fn+eps); precision=tp/(tp+fp+eps); f1=2*recall*precision/(recall+precision+eps)
iou=tp/(tp+fp+fn+eps); oa=(tp+tn)/(tn+fp+fn+tp+eps)
pre=((tp+fn)*(tp+fp)+(fp+tn)*(fn+tn))/((tn+fp+fn+tp)**2+eps)
kappa=(oa-pre)/(1-pre+eps)
metrics={"checkpoint":'/content/drive/MyDrive/SNUNet/SYSU/checkpoints/checkpoint_epoch_199.pt',"selection":"final epoch (199/200)","Kappa":float(kappa),"IoU":float(iou),"F1":float(f1),"recall":float(recall),"precision":float(precision),"OA":float(oa),"confusion_matrix":{"tn":tn,"fp":fp,"fn":fn,"tp":tp}}
(report/"test_metrics_mobile_cdnet.json").write_text(json.dumps(metrics,indent=2)+"\n")
(report/"testLog.txt").write_text("Epoch\tKappa\tIoU\tF1\tR\tP\nTest\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n" % (kappa,iou,f1,recall,precision))
scio.savemat(report/"Vis"/"results.mat",{k:v for k,v in metrics.items() if isinstance(v,float)})
print("Test :\t Kappa (te) = %.4f\t IoU (te) = %.4f\t F1 (te) = %.4f\t R (te) = %.4f\t P (te) = %.4f" % (kappa,iou,f1,recall,precision))
print("Saved:",report/"test_metrics_mobile_cdnet.json")

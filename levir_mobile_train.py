"""Train SNUNet_ECAM on LEVIRCD with the Mobile-CDNet LEVIR schedule."""
import argparse, csv, json, math, random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from models.Models import SNUNet_ECAM
import torch.nn.functional as F

MEAN = torch.tensor([0.406, 0.456, 0.485]).view(3, 1, 1)
STD = torch.tensor([0.225, 0.224, 0.229]).view(3, 1, 1)

class LEVIRCD(Dataset):
    def __init__(self, root, split, augment=False):
        self.root, self.split, self.augment = Path(root), split, augment
        list_file = self.root / split / "list" / f"{split}.txt"
        self.names = [x.strip() for x in list_file.read_text().splitlines() if x.strip()]
        if not self.names:
            raise RuntimeError(f"No samples in {list_file}")
    def __len__(self): return len(self.names)
    def __getitem__(self, i):
        name = self.names[i]; base = self.root / self.split
        a = Image.open(base/"A"/name).convert("RGB")
        b = Image.open(base/"B"/name).convert("RGB")
        y = Image.open(base/"label"/name).convert("L")
        # Mobile-CDNet scale is 256x256; LEVIRCD_256 is already 256 but enforce it.
        if a.size != (256,256):
            a=a.resize((256,256), Image.Resampling.BILINEAR); b=b.resize((256,256), Image.Resampling.BILINEAR); y=y.resize((256,256), Image.Resampling.NEAREST)
        if self.augment:
            # Equivalent paired spatial/temporal augmentation contract.
            if random.random() < .5: a,b,y=a.transpose(Image.Transpose.FLIP_LEFT_RIGHT),b.transpose(Image.Transpose.FLIP_LEFT_RIGHT),y.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < .5: a,b,y=a.transpose(Image.Transpose.FLIP_TOP_BOTTOM),b.transpose(Image.Transpose.FLIP_TOP_BOTTOM),y.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            if random.random() < .5: a,b=b,a
        def tensor(im):
            return torch.from_numpy(np.asarray(im, dtype=np.float32).transpose(2,0,1)/255.)
        a, b = (tensor(a)-MEAN)/STD, (tensor(b)-MEAN)/STD
        y = torch.from_numpy((np.asarray(y, dtype=np.uint8) > 127).astype(np.int64))
        return a, b, y, name

def loss_mobile_style(outputs, target):
    """Mobile-CDNet's foreground BCE + Dice, adapted to SNUNet's two-class logits."""
    target = target.float()
    losses = []
    for logits in outputs:
        prob = torch.softmax(logits, dim=1)[:, 1]
        bce = F.binary_cross_entropy(prob, target)
        inter = (prob * target).sum()
        dice = (2 * inter + 1e-5) / (prob.sum() + target.sum() + 1e-5)
        losses.append(bce + 1 - dice)
    return sum(losses) / len(losses)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); tp=fp=tn=fn=0; losses=[]
    for a,b,y,_ in loader:
        a,b,y=a.to(device),b.to(device),y.to(device)
        out=model(a,b); losses.append(loss_mobile_style(out,y).item())
        p=out[-1].argmax(1)
        tp += ((p==1)&(y==1)).sum().item(); fp += ((p==1)&(y==0)).sum().item()
        tn += ((p==0)&(y==0)).sum().item(); fn += ((p==0)&(y==1)).sum().item()
    precision=tp/(tp+fp+1e-8); recall=tp/(tp+fn+1e-8); f1=2*precision*recall/(precision+recall+1e-8)
    iou=tp/(tp+fp+fn+1e-8); total=tp+fp+tn+fn; po=(tp+tn)/(total+1e-8)
    pe=((tp+fp)*(tp+fn)+(fn+tn)*(fp+tn))/((total+1e-8)**2); kappa=(po-pe)/(1-pe+1e-8)
    return dict(loss=float(np.mean(losses)), precision=precision, recall=recall, f1=f1, iou=iou, kappa=kappa, tp=tp, fp=fp, tn=tn, fn=fn)

def mobile_lr(base_lr, epoch, step_loss, epoch_step):
    lr=base_lr*(0.1**(epoch//step_loss))
    if epoch==0 and epoch_step<200: lr=base_lr*(0.9*(epoch_step+1)/200+0.1)
    return lr

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/content/LEVIRCD_256")
    ap.add_argument("--output-root", default="/content/drive/MyDrive/SNUNet/LEVIR")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-steps", type=int, default=89000)
    ap.add_argument("--step-loss", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2333)
    ap.add_argument("--resume", default="")
    args=ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(args.output_root); ckpt_dir=out/"checkpoints"; log_dir=out/"logs"; ckpt_dir.mkdir(parents=True,exist_ok=True); log_dir.mkdir(parents=True,exist_ok=True)
    (out/"config.json").write_text(json.dumps(vars(args)|{"model":"SNUNet_ECAM","loss":"CE(gamma=0)+Dice (Mobile BCE+Dice analogue)","mask_colors":{"TP":"#FFFFFF","FP":"#FF0000","TN":"#000000","FN":"#00FFFF"}},indent=2))
    train_loader=DataLoader(LEVIRCD(args.data_root,"train",True),batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,pin_memory=True,drop_last=True)
    val_loader=DataLoader(LEVIRCD(args.data_root,"val",False),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers,pin_memory=True)
    model=SNUNet_ECAM(3,2).to(device); opt=torch.optim.Adam(model.parameters(),args.lr,betas=(.9,.99),eps=1e-8,weight_decay=1e-4)
    max_epochs=math.ceil(args.max_steps/len(train_loader)); start_epoch=0; best_f1=-1.
    if args.resume:
        state=torch.load(args.resume,map_location=device); model.load_state_dict(state["model"]); opt.load_state_dict(state["optimizer"]); start_epoch=state["epoch"]; best_f1=state.get("best_f1",-1.)
    with (log_dir/"train_metrics.csv").open("a",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["epoch","lr","train_loss","val_loss","precision","recall","f1","iou","kappa"]); 
        if f.tell()==0: writer.writeheader()
        for epoch in range(start_epoch,max_epochs):
            model.train(); losses=[]
            for step,(a,b,y,_) in enumerate(train_loader):
                lr=mobile_lr(args.lr,epoch,args.step_loss,step)
                for g in opt.param_groups: g["lr"]=lr
                a,b,y=a.to(device),b.to(device),y.to(device); opt.zero_grad()
                loss=loss_mobile_style(model(a,b),y); loss.backward(); opt.step(); losses.append(loss.item())
            val=evaluate(model,val_loader,device); row=dict(epoch=epoch,lr=lr,train_loss=float(np.mean(losses)),val_loss=val["loss"],precision=val["precision"],recall=val["recall"],f1=val["f1"],iou=val["iou"],kappa=val["kappa"]); writer.writerow(row); f.flush()
            state={"epoch":epoch+1,"model":model.state_dict(),"optimizer":opt.state_dict(),"best_f1":max(best_f1,val["f1"]),"args":vars(args),"val":val}
            torch.save(state,ckpt_dir/"last_checkpoint.pt")
            if val["f1"]>=best_f1: best_f1=val["f1"]; torch.save(state,ckpt_dir/"best_model.pt")
            print(row)
if __name__=="__main__": main()

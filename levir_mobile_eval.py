"""Evaluate a SNUNet LEVIR checkpoint and save TP/FP/TN/FN color masks."""
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from levir_mobile_train import LEVIRCD, loss_mobile_style
from models.Models import SNUNet_ECAM

COLORS={"TP":(255,255,255),"FP":(255,0,0),"TN":(0,0,0),"FN":(0,255,255)}
@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data-root",default="/content/LEVIRCD_256"); ap.add_argument("--output-root",default="/content/drive/MyDrive/SNUNet/LEVIR")
    ap.add_argument("--checkpoint",default=""); ap.add_argument("--batch-size",type=int,default=1); ap.add_argument("--num-workers",type=int,default=2)
    args=ap.parse_args(); out=Path(args.output_root); ckpt=Path(args.checkpoint) if args.checkpoint else out/"checkpoints"/"best_model.pt"
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); state=torch.load(ckpt,map_location=device)
    model=SNUNet_ECAM(3,2).to(device); model.load_state_dict(state["model"]); model.eval()
    mask_dir=out/"results"/"confusion_masks"; mask_dir.mkdir(parents=True,exist_ok=True)
    loader=DataLoader(LEVIRCD(args.data_root,"test",False),batch_size=args.batch_size,shuffle=False,num_workers=args.num_workers)
    tp=fp=tn=fn=0; losses=[]
    for a,b,y,names in loader:
        outp=model(a.to(device),b.to(device)); losses.append(loss_mobile_style(outp,y.to(device)).item()); p=outp[-1].argmax(1).cpu().numpy(); y=y.numpy()
        for pred,gt,name in zip(p,y,names):
            rgb=np.zeros((*gt.shape,3),np.uint8); rgb[(pred==1)&(gt==1)]=COLORS["TP"]; rgb[(pred==1)&(gt==0)]=COLORS["FP"]; rgb[(pred==0)&(gt==0)]=COLORS["TN"]; rgb[(pred==0)&(gt==1)]=COLORS["FN"]
            Image.fromarray(rgb).save(mask_dir/name)
            tp+=int(((pred==1)&(gt==1)).sum()); fp+=int(((pred==1)&(gt==0)).sum()); tn+=int(((pred==0)&(gt==0)).sum()); fn+=int(((pred==0)&(gt==1)).sum())
    precision=tp/(tp+fp+1e-8); recall=tp/(tp+fn+1e-8); f1=2*precision*recall/(precision+recall+1e-8); iou=tp/(tp+fp+fn+1e-8)
    metrics=dict(loss=float(np.mean(losses)),precision=precision,recall=recall,f1=f1,iou=iou,tp=tp,fp=fp,tn=tn,fn=fn,colors=COLORS,checkpoint=str(ckpt))
    (out/"results").mkdir(parents=True,exist_ok=True); (out/"results"/"test_metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2)); print("masks:",mask_dir)
if __name__=="__main__": main()

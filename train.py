import os
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from SRCNN import SRCNN
from dataloader import load_loader
from utils import accuracy_metric, plot_training_process
import wandb


def parse():
    parser = argparse.ArgumentParser(description='SRCNN Training')
    parser.add_argument('--train_data', default="data/data_train/train", help='train data path', type=str)
    parser.add_argument('--val_data', default="data/data_train/val", help='val data path', type=str)
    parser.add_argument('--save_path', default="models", help='models path', type=str)
    parser.add_argument('--num_epochs', default=2000 , help='number of epochs', type=int)
    parser.add_argument('--key', default=None, help='wandb key', type=str)
    parser.add_argument('--pre_train', default=None, help='pretrained model', type=str)
    args = parser.parse_args()
    return args



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    
    loop = tqdm(train_loader)
    
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0

    for i,(imgs, gts) in enumerate(loop):
        imgs = imgs.to(device)
        gts = gts.to(device)
        
        with torch.set_grad_enabled(True):
            # forward propagation
            preds = model(imgs)
            # calculate loss
            loss=criterion(preds, gts)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calculate accuracy
        psnr, ssim = accuracy_metric(gts, preds)

        # update status
        loop.set_postfix({"loss":loss.item(), 
                          "PSNR": psnr.item(),
                          "SSIM": ssim.item()})

        epoch_loss+=loss.item()
        epoch_psnr+=psnr.item()
        epoch_ssim+=ssim.item()

        # if i%100==0 and i>0:
        #     print(f"Iteration {i}: Train acc {epoch_acc/i} - Train loss{epoch_loss/i}")

    epoch_loss/=len(train_loader)
    epoch_psnr/=len(train_loader)
    epoch_ssim/=len(train_loader)

    model.eval()
    return model, epoch_loss, epoch_psnr, epoch_ssim



def validation(model, val_loader, criterion, device):
    model.eval()
    
    loop = tqdm(val_loader)
    
    epoch_loss = 0
    epoch_psnr = 0
    epoch_ssim = 0

    for i,(imgs, gts) in enumerate(loop):
        imgs = imgs.to(device)
        gts = gts.to(device)
        
        with torch.set_grad_enabled(True):
            # forward propagation
            preds = model(imgs)
            # calculate loss
            loss=criterion(preds, gts)

        # calculate accuracy
        psnr, ssim = accuracy_metric(gts, preds)

        # update status
        loop.set_postfix({"loss":loss.item(), 
                          "PSNR": psnr.item(),
                          "SSIM": ssim.item()})

        epoch_loss+=loss.item()
        epoch_psnr+=psnr.item()
        epoch_ssim+=ssim.item()

    epoch_loss/=len(val_loader)
    epoch_psnr/=len(val_loader)
    epoch_ssim/=len(val_loader)

    model.eval()
    return model, epoch_loss, epoch_psnr, epoch_ssim


def main():
    args = parse()

    # login wandb
    if args.key is not None:
        wandb.login(key=args.key)
    else:
        wandb.login()

    # torch.backends.cudnn.enabled=False
    device=torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    
    # train parameter
    config = dict(
                    save_path = args.save_path,
                    num_epoch = args.num_epochs,
                    batch_size = 1,
                    num_worker = 0,
                    lr = 1e-5,
                    weight_decay = 1e-4,
                    dataset = "T19",
                    architecture = "SRCNN",
                    device = device
                 )

    # wandb config
    wandb.init(project="SRCNN Training", config=config)

    # declare model
    srcnn = SRCNN()
    if args.pre_train is not None:
        srcnn.load_state_dict(torch.load(args.pre_train, map_location="cpu")["model_state_dict"])
    srcnn = srcnn.to(device)
    # declare loss function
    criterion=nn.MSELoss()
    # declare optimizer
    # optimizer = optim.SGD(mcnn.parameters(), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])
    optimizer = optim.Adam(srcnn.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    # declare scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
    
    # generate dataloader
    train_loader, val_loader = load_loader(
                                            train_folder=args.train_data,
                                            val_folder=args.val_data,
                                            img_size=33,
                                            batch_size=32,
                                            num_workers=2,
                                            scale=2
                                            )
    
    min_err = 10000000000000
    max_psnr = -1
    max_ssim = -1
    
    train_info = {  "epoch": [],
                    "train_loss": [],
                    "train_psnr": [],
                    "train_ssim": [],
                    "val_loss": [],
                    "val_psnr": [],
                    "val_ssim": []
                }

    for epoch in range(0, config["num_epoch"]):
        print(f"Epoch {epoch+1}/{config['num_epoch']}...")
        # train
        srcnn, train_loss, train_psnr, train_ssim = train(srcnn, train_loader, criterion, optimizer, device)
        print(f"Train PSNR: {train_psnr} - SSIM: {train_ssim} - Loss: {train_loss}")

        # eval
        srcnn, val_loss, val_psnr, val_ssim = validation(srcnn, val_loader, criterion, device)
        print(f"Val PSNR: {val_psnr} - SSIM: {val_ssim} - Loss: {val_loss}")

        # update wandb log
        wandb.log({"train_loss": train_loss,
                   "train_psnr": train_psnr,
                   "train_ssim": train_ssim,
                   "val_loss": val_loss,
                   "val_psnr": val_psnr,
                   "val_ssim": val_ssim
                   }, step=epoch)
        
        # save
        save_path = config["save_path"]
        if min_err > train_loss and max_psnr < val_psnr and max_ssim < val_ssim:
            min_err = train_loss
            max_psnr = val_psnr
            max_ssim = val_ssim
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': srcnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'psnr': val_psnr,
                        'ssim': val_ssim
                        }, f"{save_path}/models/best.pth.tar")
            print("Saved BEST model!")
        
        torch.save({
                        'epoch': epoch,
                        'model_state_dict': srcnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_loss,
                        'psnr': val_psnr,
                        'ssim': val_ssim
                        }, f"{save_path}/models/last.pth.tar")

        # update status
        train_info["epoch"].append(epoch)
        train_info["train_psnr"].append(train_psnr)
        train_info["train_ssim"].append(train_ssim)
        train_info["train_loss"].append(train_loss)
        train_info["val_psnr"].append(val_psnr)
        train_info["val_ssim"].append(val_ssim)
        train_info["val_loss"].append(val_loss)
        
        scheduler.step()

    plot_training_process(train_info=train_info, save_path=config["save_path"])
    print("Finished training!")



if __name__ == "__main__":
    main()

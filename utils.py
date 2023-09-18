import matplotlib.pyplot as plt
from skimage.measure import compare_psnr, compare_ssim

def accuracy_metric(gts, preds):
    psnr_avg = 0
    ssim_avg = 0
    for gt, pred in zip(gts, preds):
        psnr = compare_psnr(gt, pred)
        ssim = compare_ssim(gt, pred, multichannel=True)
        psnr_avg+=psnr
        ssim_avg+=ssim
    psnr_avg/=gts.shape[0]
    ssim_avg/=preds.shape[0]
    return psnr_avg, ssim_avg




def plot_training_process(train_info, save_path):
    plt.plot(train_info["epoch"], train_info["train_loss"])
    plt.title("Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{save_path}/train_loss.png")
    plt.close()

    plt.plot(train_info["epoch"], train_info["train_psnr"])
    plt.title("Train PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.savefig(f"{save_path}/train_psnr.png")
    plt.close()

    plt.plot(train_info["epoch"], train_info["train_ssim"])
    plt.title("Train SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.savefig(f"{save_path}/train_ssim.png")
    plt.close()

    plt.plot(train_info["epoch"], train_info["val_loss"])
    plt.title("Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(f"{save_path}/val_loss.png")
    plt.close()

    plt.plot(train_info["epoch"], train_info["val_psnr"])
    plt.title("Validation PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.grid(True)
    plt.savefig(f"{save_path}/val_psnr.png")
    plt.close()

    plt.plot(train_info["epoch"], train_info["val_ssim"])
    plt.title("Validation SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.grid(True)
    plt.savefig(f"{save_path}/val_ssim.png")
    plt.close()
    
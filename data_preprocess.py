from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import random

def preprocess(img:np.ndarray, stride=14, size=33):
    img_list = []
    h, w, _ = img.shape
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            subimg = img[i:i+size, j:j+size].copy()
            if subimg.shape[0] == subimg.shape[1] or subimg.shape[0]==size: 
                img_list.append(subimg)
    return img_list

def show_sample(data_folder, num_samples):
    img_path_list = glob(f"{data_folder}/*.*")
    img_path_show = random.sample(img_path_list, k=num_samples)
    Nr = Nc = math.ceil(math.sqrt(num_samples))
    fig, axs = plt.subplots(Nr, Nc)
    fig.suptitle('Sample images')
    idx = 0
    for i in range(Nr):
        for j in range(Nc):
            img = cv2.imread(img_path_show[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].label_outer()
            axs[i, j].axis('off')
            idx+=1
    plt.show()

if __name__ == "__main__":
    data_path_list = glob("./data/T91/*.*")
    data_preprocess_folder = "./data/T91_preprocess"

    idx = 0
    for path in tqdm(data_path_list):
        img = cv2.imread(path)
        preprocess_img_list = preprocess(img)
        for pre_img in preprocess_img_list:
            cv2.imwrite(data_preprocess_folder+f"/{str(idx).zfill(6)}.png", pre_img)
            idx+=1

    final_data_path_list = glob("./data/T91_preprocess/*.*")
    print(len(final_data_path_list))
    print("Finished!")
    show_sample(data_preprocess_folder, num_samples=25)

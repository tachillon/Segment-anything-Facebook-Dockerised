import os
import argparse
import numpy as np
import torch 
import torchvision
import time
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

print("PyTorch version:     ", torch.__version__)
print("Torchvision version: ", torchvision.__version__)
print("CUDA is available:   ", torch.cuda.is_available())

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))


def draw_anns(masks, image, output_path="result.png"):
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(output_path)


argparser = argparse.ArgumentParser()
argparser.add_argument("--image_path", type=str, default="1.jpg")
argparser.add_argument("--checkpoint", type=str, default="/home/sam_vit_h_4b8939.pth")
args = argparser.parse_args()

# Delete the folder if it exists
if os.path.exists("results"):
    os.system("rm -rf " + "results")
# Create the folder again
os.makedirs("results")

sam_checkpoint = args.checkpoint
image_path     = args.image_path
image          = cv2.imread(image_path)
image          = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_bgr      = cv2.imread(image_path)

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_type = "default"
sam        = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

start = time.time()
masks = mask_generator.generate(image)
end = time.time()
print("Elapsed time (inference): {} secs".format(round(end - start, 2)))

start = time.time()
draw_anns(masks, image, output_path="results/result.png")
end = time.time()
print("Elapsed time (draw annotations): {} secs".format(round(end - start, 2)))
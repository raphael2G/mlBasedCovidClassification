import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchsummary import summary

from torchvision import datasets, transforms

from linformer import Linformer
from vit_pytorch.efficient import ViT

from PIL import Image


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# - - - - - - intializing model - - - - - -

efficient_transformer = Linformer(
    dim=1024,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

pt_model = ViT(
    dim=1024,
    image_size=224,
    patch_size=32,
    num_classes=2,
    transformer=efficient_transformer,
    channels=3,
).to(device)



# - - - - - - load pretrained model - - - - - -
PATH = 'savedModels/ViT_covid_classifier.pt'
pretrained_net = torch.load(PATH, map_location=torch.device('cpu'))
pt_model.load_state_dict(pretrained_net)


# - - - - - - add softmax activation function - - - - 
# model = nn.Sequential(
#     model,
#     nn.Softmax(dim=1) 
# )

summary(pt_model, (3, 224, 224))


# - - - - - - process data - - - - - - 
def get_data_from_file(img_path, get_label=False):
    img = Image.open(img_path)

    if get_label != False: 
        label = img_path.split("/")[-1].split("_")[0]
        return img, label

    return img

def process_img(img):
    test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    img_tensor = test_transforms(img).resize_(1, 3, 224, 224)
    return img_tensor


def test(IMG_PATH):
    img, label = get_data_from_file(IMG_PATH, get_label=True)
    print('- - - - - - ' + label + ' - - - - - - ')
    processed_img = process_img(img)

    outputs = pt_model(processed_img).detach().numpy()[0]

    print(outputs)
    print(f'CONFIDENCE: {outputs.max():.4f}%')
    print('COVID DETECTED') if outputs[1] > outputs[0] else print('COVID NOT DETECTED')

test('data/noncovid_ct_scan_99-66.jpeg')
test('data/covid_ct_scan_0-14.jpg')
test('data/covid_ct_scan_0-27L.jpg')






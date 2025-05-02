import cv2
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.classification import MultilabelAveragePrecision

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from model.RTwithoutFFIN_model import *
from model.RT_Rand_model import *
from model.RT_CLIP_model import *
from model.TSTwithoutFFIN_model import *
from model.TST_Rand_model import *
from model.TST_CLIP_model import *
from model.CFNm_model import *
from dataset import *

def myimshows(imgs, titles=False, fname="test.jpg", size=6):
    lens = len(imgs)
    fig = plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap='Reds')
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(fname, bbox_inches='tight')
    plt.show()

def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr

def reshape_transform(tensor, height=14, width=14, frame=16, rgb_img=None):
    #tensor = tensor.permute(1, 0, 2)    # b,1+t*h*w,d
    #tensor = tensor[0]

    tensor = tensor[1]
    result = tensor[:, 15, 1:].reshape(tensor.size(0), frame, height, width)
    result = result[0, 3, :, :].detach().numpy()
    result = cv2.resize(result, (224, 224), interpolation=cv2.INTER_LINEAR)
    result = (result - np.min(result)) / (np.max(result) - np.min(result))
    cam_image = show_cam_on_image(rgb_img, result, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    os.makedirs(r'cam', exist_ok=True)
    cam_output_path = os.path.join('decoder_cam_l15_320.jpg')
    cv2.imwrite(cam_output_path, cam_image)

    result = tensor[:, 1:, :].reshape(tensor.size(0), frame, height, width, tensor.size(2))
    result = result[:, 0, :, :, :]

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

parser = argparse.ArgumentParser(description='MBBRN_cam')
parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--device", default='cuda:5', type=str, help="0, 1, 2, 3, 4, 5, 6, 7")

# RT模型相关参数
parser.add_argument("--RT_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--RT_pt", default='path/RT_best_stage', help="Load the trained model, Default: path or None")

# TST模型相关参数
parser.add_argument("--TST_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--TST_pt", default='path/TST_best_stage', help="Load the trained model, Default: path or None")

# MBBRN模型相关参数
parser.add_argument("--MBBRN_pt", default='path/best_16_768', type=str, help="Load the trained model, Default: path or None")
parser.add_argument("--frames", default=16, type=int, help="Number of frames in a video")
parser.add_argument('--dataroot', default=r'path/AnimalKingdom', type=str, help='dataset root')
opt = parser.parse_args()

def collate_fn(batch):
    images, data, animal_action = zip(*batch)
    images = list(images)
    data = torch.stack(data)
    return images, data, animal_action

# define dataset
action_class_list, animal_class_list = get_dict()
action_class_list = list(action_class_list.keys())
print("num_action_class_list = ", len(action_class_list))

set = AnimalKingdom_action_cam(root=opt.dataroot, type='train', total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
fixed_indices = [32]
subset = torch.utils.data.Subset(set, fixed_indices)
# sampler = torch.utils.data.RandomSampler(set, num_samples=10)
loader = torch.utils.data.DataLoader(
    dataset=subset,
    batch_size=1,
    collate_fn=collate_fn,
    shuffle=False
)
data_iter = iter(loader)

# define model
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

RT_model = RTwithoutFFIN(num_queries=len(action_class_list), frames=opt.frames, device=device).to(device)
TST_model = TSTwithoutFFIN(num_frames=opt.frames, num_queries=len(action_class_list), device=device).to(device)
MBBRN_model = CFNm(RT_model=RT_model, TST_model=TST_model, num_queries=len(action_class_list), device=device).to(device)
MBBRN_model.load_state_dict(torch.load(opt.MBBRN_pt, map_location=device), strict=False)
MBBRN_model.eval()

# 加载数据
images, data, action_label = next(data_iter)
image = images[0][8]
image_np = np.array(image)
data = data.to(device)
pred = MBBRN_model(data)
pred = pred.to('cpu')

label = np.where(action_label[0] == 1.)[0]
print("label: ", label)
action_probas = (torch.sigmoid(pred) > 0.5).nonzero(as_tuple=True)[1]
print("pred: ", action_probas)

pred_list = torch.sigmoid(pred).squeeze(0).detach().numpy()
top_list = np.argsort(pred_list)[-5:][::-1]
pred_list = [pred_list[i] for i in top_list]

action_class_list, _ = get_dict()
action_class_list = list(action_class_list.keys())
action_class_list = [action_class_list[i] for i in top_list]

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].imshow(image_np)
ax[0].axis('off')

ax[1].barh(action_class_list, pred_list, color='skyblue')
ax[1].set_xlim(0, 1)
ax[1].set_xlabel('Probability', fontsize=15)
ax[1].tick_params(axis='y', labelsize=15)

plt.tight_layout()
plt.savefig('path/example.png')



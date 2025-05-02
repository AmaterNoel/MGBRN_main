import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.classification import MultilabelAveragePrecision

from model.RTwithoutFFIN_model import *
from model.RT_Rand_model import *
from model.RT_CLIP_model import *
from model.TSTwithoutFFIN_model import *
from model.TST_Rand_model import *
from model.TST_CLIP_model import *
from model.CFNm_model import *
from dataset import *

parser = argparse.ArgumentParser(description='MBBRN')
parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--device", default='cuda:3', type=str, help="0, 1, 2, 3, 4, 5, 6, 7")

# RT模型相关参数
parser.add_argument("--RT_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--RT_pt", default='path/RT_best_stage6', help="Load the trained model, Default: path or None")

# TST模型相关参数
parser.add_argument("--TST_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--TST_pt", default='path/TST_best_stage', help="Load the trained model, Default: path or None")

# CFN模型相关参数
parser.add_argument("--CFNm_pt", default='path/best_16_768', type=str, help="Load the trained model, Default: path or None")
parser.add_argument("--save_last_pt", default='path', help="The address where the model is stored after training is completed, Default: path or None")
parser.add_argument("--save_best_pt", default='path', help="The address where the model is stored after training is completed, Default: path or None")
parser.add_argument("--total_epoch", default=400, type=int, help="epoch")
parser.add_argument("--frames", default=16, type=int, help="Number of frames in a video")
parser.add_argument("--batch_size", default=1, type=int, help="Size of the mini-batch")
parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
parser.add_argument('--dataroot', default=r'path/AnimalKingdom', type=str, help='dataset root')
opt = parser.parse_args()

# define dataset
action_class_list, animal_class_list = get_dict()
action_class_list = list(action_class_list.keys())
print("num_action_class_list = ", len(action_class_list))

test_set = AnimalKingdom_action(root=opt.dataroot, type='test', total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
test_sampler = torch.utils.data.RandomSampler(test_set)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=opt.batch_size,
    sampler=test_sampler,
    shuffle=False
)

text_embed = get_text_features(action_class_list)
text_embed = F.adaptive_avg_pool1d(text_embed.unsqueeze(0), 256).squeeze(0)
device = torch.device(opt.device if torch.cuda.is_available() else "cpu")

# define RT model
if opt.RT_pt == 'None':
    raise ValueError("RT_pt is not allowed to be None")
if opt.RT_type == "withoutFFIN":
    RT_model = RTwithoutFFIN(num_queries=len(action_class_list), frames=opt.frames, device=device).to(device)
    #RT_model.load_state_dict(torch.load(opt.RT_pt, map_location=device), strict=False)
elif opt.RT_type == "Rand":
    RT_model = RT_Rand_NH(num_queries=len(action_class_list), device=device).to(device)
    RT_model.load_state_dict(torch.load(opt.RT_pt, map_location=device), strict=False)
elif opt.RT_type == "CLIP":
    RT_model = RT_CLIP_NH(text_embed=text_embed, num_queries=len(action_class_list), device=device).to(device)
    RT_model.load_state_dict(torch.load(opt.RT_pt, map_location=device), strict=False)
else:
     raise ValueError("RT_type is wrong!")
#for param in RT_model.parameters():
#    param.requires_grad = False

# define TST model
if opt.TST_pt == 'None':
    raise ValueError("TST_pt is not allowed to be None")
if opt.TST_type == "withoutFFIN":
    TST_model = TSTwithoutFFIN(num_frames=opt.frames, num_queries=len(action_class_list), device=device).to(device)
    #TST_model.load_state_dict(torch.load(opt.TST_pt, map_location=device), strict=False)
elif opt.TST_type == "Rand":
    TST_model = TST_Rand_NH(num_frames=opt.frames, num_queries=len(action_class_list), device=device).to(device)
    TST_model.load_state_dict(torch.load(opt.TST_pt, map_location=device))
elif opt.TST_type == "CLIP":
    TST_model = TST_CLIP_NH(text_embed=text_embed, num_frames=opt.frames, num_queries=len(action_class_list), device=device).to(device)
    TST_model.load_state_dict(torch.load(opt.TST_pt, map_location=device))
else:
    raise ValueError("TST_type is wrong!")
#for param in TST_model.parameters():
#    param.requires_grad = False

# define CFN model
CFN_model = CFNm(RT_model=RT_model, TST_model=TST_model, num_queries=len(action_class_list), device=device).to(device)
if opt.CFNm_pt != 'None':
    CFN_model.load_state_dict(torch.load(opt.CFNm_pt, map_location=device), strict=False)

action_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
print('Parameter Space: ABS: {:.2f}'.format(count_parameters(CFN_model) / 268435456) + " GB")

# 可视化
action_mAP = []
start_epoch = int(opt.save_last_pt[-1])

class_names = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4, 'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8, 'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12, 'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17, 'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21, 'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25, 'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28, 'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31, 'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34, 'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39, 'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43, 'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48, 'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52, 'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56, 'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60, 'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64, 'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69, 'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73, 'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77, 'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81, 'Performing allo-grooming': 82, 'Performing allo-preening': 83, 'Performing copulatory mounting': 84, 'Performing sexual exploration': 85, 'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89, 'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94, 'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99, 'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103, 'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107, 'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111, 'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115, 'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119, 'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123, 'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127, 'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131, 'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136, 'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
class_names = {v: k for k, v in class_names.items()}
#class_names = {63: "Holding hands", 65: "Hugging", 106: "Showing affection", 1: "Attacking", 14: "Chasing", 17: "Coiling", 18: "Competing for dominance", 47: "Fighting", 91: "Preying", 138: "Wrapping prey", 112: "Spitting venom", 3: "Barking", 10: "Calling", 56: "Giving off light", 136: "Waving", 21: "Dead", 39: "Dying", 11: "Camouflaging", 23: "Defensive rearing", 26: "Displaying defensive pose", 42: "Escaping", 96: "Retaliating", 55: "Giving birth", 60: "Hatching", 71: "Laying eggs", 77: "Molting", 19: "Dancing", 81: "Performing sexual display", 84: "Performing copulatory mounting", 86: "Performing sexual pursuit", 87: "Playing", 124: "Swimming in circles", 7: "Being eaten", 88: "Playing dead", 127: "Trapped"}
class_stats = {cls: {'i': cls, 'c': 0, 't': 0, 'name': name} for cls, name in class_names.items()}

# define eval
CFN_model.eval()
action_loss_meter = AverageMeter()
RT_loss_meter = AverageMeter()
TST_loss_meter = AverageMeter()
for data, action_label in test_loader:
    data = data.to(device)
    action_label = action_label.long()

    with torch.no_grad():
        action_pred = CFN_model(data)
    action_pred = action_pred.to('cpu')
    action_eval = action_eval_metric(action_pred, action_label)
    action_loss_meter.update(action_eval.item(), data.shape[0])

    label = np.where(action_label[0] == 1.)[0]
    action_probas = (torch.sigmoid(action_pred) > 0.5).nonzero(as_tuple=True)[1]
    for cls, stats in class_stats.items():
        # 检查 label 中是否包含 cls
        if (label == cls).any():  # 如果 label 中存在 cls
            stats['t'] += 1  # 更新总样本数

            # 检查 action_probas 中是否包含 cls
            if (action_probas == cls).any():  # 如果 action_probas 中也存在 cls
                stats['c'] += 1  # 更新正确预测数

def calculate_accuracy(correct, total):
    return correct / total * 100 if total > 0 else 0.0

for cls, stats in class_stats.items():
    class_accuracy = calculate_accuracy(stats['c'], stats['t'])
    print(f"[INFO] {stats['name']} (Index: {stats['i']}):")
    print(f"  Correct Predictions: {stats['c']} / Total Samples: {stats['t']}")
    print(f"  Accuracy: {class_accuracy:.2f}%\n")

action_mAP.append(action_loss_meter.avg * 100)
print("[INFO] Action Evaluation Metric: {:.4f}".format(action_loss_meter.avg * 100) + "\n", flush=True)

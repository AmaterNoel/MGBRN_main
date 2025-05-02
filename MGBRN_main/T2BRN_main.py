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
from model.T2_model import *
from dataset import *

parser = argparse.ArgumentParser(description='MBBRN')
parser.add_argument("--seed", default=1, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
parser.add_argument("--device", default='cuda:3', type=str, help="0, 1, 2, 3, 4, 5, 6, 7")

# RT模型相关参数
parser.add_argument("--RT_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--RT_pt", default='path/RT_best_stage', help="Load the trained model, Default: path or None")

# TST模型相关参数
parser.add_argument("--TST_type", default='withoutFFIN', type=str, help="withoutFFIN, Rand, CLIP")
parser.add_argument("--TST_pt", default='path/TST_best_stage', help="Load the trained model, Default: path or None")

# CFN模型相关参数
parser.add_argument("--CFNm_pt", default='None', type=str, help="Load the trained model, Default: path or None")
parser.add_argument("--save_last_pt", default='path/T2_last_768', help="The address where the model is stored after training is completed, Default: path or None")
parser.add_argument("--save_best_pt", default='path/T2_best_768', help="The address where the model is stored after training is completed, Default: path or None")
parser.add_argument("--total_epoch", default=400, type=int, help="epoch")
parser.add_argument("--frames", default=8, type=int, help="Number of frames in a video")
parser.add_argument("--batch_size", default=8, type=int, help="Size of the mini-batch")
parser.add_argument("--test_every", default=5, type=int, help="Test the model every this number of epochs")
parser.add_argument('--dataroot', default=r'path/AnimalKingdom', type=str, help='dataset root')
opt = parser.parse_args()

# define dataset
action_class_list, animal_class_list = get_dict()
action_class_list = list(action_class_list.keys())
print("num_action_class_list = ", len(action_class_list))

train_set = AnimalKingdom_action(root=opt.dataroot, type='train', total_length=opt.frames, action_class_list=action_class_list, random_shift=False)
train_sampler = torch.utils.data.RandomSampler(train_set, num_samples=2500)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=opt.batch_size,
    sampler=train_sampler,
    shuffle=False
)

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
    RT_model = TSTwithoutFFIN(num_frames=opt.frames, num_queries=len(action_class_list), device=device).to(device)
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
    CFN_model.load_state_dict(torch.load(opt.CFN_pt, map_location=device), strict=False)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(CFN_model.parameters(), lr=0.00001)
scheduler = optim.lr_scheduler.StepLR(optimizer, 200)
action_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
RT_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
TST_eval_metric = MultilabelAveragePrecision(num_labels=len(action_class_list), average='micro')
print('Parameter Space: ABS: {:.2f}'.format(count_parameters(CFN_model) / 268435456) + " GB")

# 可视化
action_mAP = []
start_epoch = int(opt.save_last_pt[-1])
x = list(range((start_epoch - 1) * opt.total_epoch + opt.test_every, start_epoch * opt.total_epoch+1, opt.test_every))

# define train
best_mAP = 0
best_pt = None
for index in range(opt.total_epoch):
    CFN_model.train()
    action_loss_meter = AverageMeter()
    start_time = time.time()
    for data, action_label in train_loader:
        data = data.to(device)
        action_label = action_label.to(device)

        optimizer.zero_grad()
        action_pred = CFN_model(data)

        action_loss = criterion(action_pred, action_label)
        action_loss.backward()
        optimizer.step()

        action_loss_meter.update(action_loss, data.shape[0])
    elapsed_time = time.time() - start_time
    scheduler.step()
    print("Epoch [" + str(index + 1) + "]"
          + "[" + str(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))) + "]"
          + " action_loss: " + "{:.4f}".format(action_loss_meter.avg), flush=True)

    if (index + 1) % opt.test_every == 0:
        CFN_model.eval()
        action_loss_meter = AverageMeter()
        RT_loss_meter = AverageMeter()
        TST_loss_meter = AverageMeter()
        for data, action_label in test_loader:
            data = data.to(device)
            action_label = action_label.long().to(device)

            with torch.no_grad():
                action_pred = CFN_model(data)
            action_eval = action_eval_metric(action_pred, action_label)

            #plt.figure()
            #plt.plot(RT_pred[0].cpu().numpy(), label='RT_pred')
            #plt.plot(TST_pred[0].cpu().numpy(), label='TST_pred')
            #plt.plot(action_pred[0].cpu().numpy(), label='action_pred')
            #plt.plot(action_label[0].cpu().numpy(), label='ground truth')
            #plt.legend()
            #save_path = "./eval_data/CFNm_eval/eval_" + str((start_epoch-1)*opt.total_epoch+index+1) + ".png"
            #plt.savefig(save_path)
            #plt.close()

            action_loss_meter.update(action_eval.item(), data.shape[0])
        action_mAP.append(action_loss_meter.avg * 100)
        print("[INFO] Action Evaluation Metric: {:.4f}".format(action_loss_meter.avg * 100) + "\n", flush=True)
        if action_loss_meter.avg * 100 > best_mAP:
            best_mAP = action_loss_meter.avg * 100
            state_dict = CFN_model.state_dict()
            FFIN_state_dict = {k: v for k, v in state_dict.items() if "RT_model" not in k and "TST_model" not in k}
            best_pt = FFIN_state_dict

    if (index + 1) % 50 == 0:
        # 模型保存
        state_dict = CFN_model.state_dict()
        # FFIN_state_dict = {k: v for k, v in state_dict.items() if "RT_model" not in k and "TST_model" not in k}
        torch.save(state_dict, opt.save_last_pt)
        torch.save(state_dict, opt.save_best_pt)
        print("[INFO] Best Action Evaluation Metric: {:.4f}".format(best_mAP), flush=True)

# 训练效果可视化
plt.plot(x, action_mAP, label='action_mAP')
plt.legend()
plt.savefig(r'path/eval_data' + str((start_epoch-1)*opt.total_epoch) + '-' + str(start_epoch*opt.total_epoch) + '.png')

# 模型保存
state_dict = CFN_model.state_dict()
#FFIN_state_dict = {k: v for k, v in state_dict.items() if "RT_model" not in k and "TST_model" not in k}
torch.save(state_dict, opt.save_last_pt)
torch.save(state_dict, opt.save_best_pt)
print("[INFO] Best Action Evaluation Metric: {:.4f}".format(best_mAP), flush=True)
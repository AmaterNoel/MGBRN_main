import os
import csv
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import compress
import torchvision.transforms as T
from torchvision.transforms import Compose
from torch.utils.data.dataset import Dataset
from PIL import Image, ImageOps, ImageFilter
from transformers import CLIPTokenizer, CLIPTextModel

def get_dict():
    action_dict = {'Holding some clothes': 0, 'Putting clothes somewhere': 1, 'Taking some clothes from somewhere': 2,
                     'Throwing clothes somewhere': 3, 'Tidying some clothes': 4, 'Washing some clothes': 5,
                     'Closing a door': 6, 'Fixing a door': 7, 'Opening a door': 8, 'Putting something on a table': 9,
                     'Sitting on a table': 10, 'Sitting at a table': 11, 'Tidying up a table': 12,
                     'Washing a table': 13, 'Working at a table': 14, 'Holding a phone/camera': 15,
                     'Playing with a phone/camera': 16, 'Putting a phone/camera somewhere': 17,
                     'Taking a phone/camera from somewhere': 18, 'Talking on a phone/camera': 19, 'Holding a bag': 20,
                     'Opening a bag': 21, 'Putting a bag somewhere': 22, 'Taking a bag from somewhere': 23,
                     'Throwing a bag somewhere': 24, 'Closing a book': 25, 'Holding a book': 26, 'Opening a book': 27,
                     'Putting a book somewhere': 28, 'Smiling at a book': 29, 'Taking a book from somewhere': 30,
                     'Throwing a book somewhere': 31, 'Watching/Reading/Looking at a book': 32, 'Holding a towel/s': 33,
                     'Putting a towel/s somewhere': 34, 'Taking a towel/s from somewhere': 35,
                     'Throwing a towel/s somewhere': 36, 'Tidying up a towel/s': 37,
                     'Washing something with a towel': 38, 'Closing a box': 39, 'Holding a box': 40,
                     'Opening a box': 41, 'Putting a box somewhere': 42, 'Taking a box from somewhere': 43,
                     'Taking something from a box': 44, 'Throwing a box somewhere': 45, 'Closing a laptop': 46,
                     'Holding a laptop': 47, 'Opening a laptop': 48, 'Putting a laptop somewhere': 49,
                     'Taking a laptop from somewhere': 50, 'Watching a laptop or something on a laptop': 51,
                     'Working/Playing on a laptop': 52, 'Holding a shoe/shoes': 53, 'Putting shoes somewhere': 54,
                     'Putting on shoe/shoes': 55, 'Taking shoes from somewhere': 56, 'Taking off some shoes': 57,
                     'Throwing shoes somewhere': 58, 'Sitting in a chair': 59, 'Standing on a chair': 60,
                     'Holding some food': 61, 'Putting some food somewhere': 62, 'Taking food from somewhere': 63,
                     'Throwing food somewhere': 64, 'Eating a sandwich': 65, 'Making a sandwich': 66,
                     'Holding a sandwich': 67, 'Putting a sandwich somewhere': 68,
                     'Taking a sandwich from somewhere': 69, 'Holding a blanket': 70, 'Putting a blanket somewhere': 71,
                     'Snuggling with a blanket': 72, 'Taking a blanket from somewhere': 73,
                     'Throwing a blanket somewhere': 74, 'Tidying up a blanket/s': 75, 'Holding a pillow': 76,
                     'Putting a pillow somewhere': 77, 'Snuggling with a pillow': 78,
                     'Taking a pillow from somewhere': 79, 'Throwing a pillow somewhere': 80,
                     'Putting something on a shelf': 81, 'Tidying a shelf or something on a shelf': 82,
                     'Reaching for and grabbing a picture': 83, 'Holding a picture': 84, 'Laughing at a picture': 85,
                     'Putting a picture somewhere': 86, 'Taking a picture of something': 87,
                     'Watching/looking at a picture': 88, 'Closing a window': 89, 'Opening a window': 90,
                     'Washing a window': 91, 'Watching/Looking outside of a window': 92, 'Holding a mirror': 93,
                     'Smiling in a mirror': 94, 'Washing a mirror': 95,
                     'Watching something/someone/themselves in a mirror': 96, 'Walking through a doorway': 97,
                     'Holding a broom': 98, 'Putting a broom somewhere': 99, 'Taking a broom from somewhere': 100,
                     'Throwing a broom somewhere': 101, 'Tidying up with a broom': 102, 'Fixing a light': 103,
                     'Turning on a light': 104, 'Turning off a light': 105, 'Drinking from a cup/glass/bottle': 106,
                     'Holding a cup/glass/bottle of something': 107, 'Pouring something into a cup/glass/bottle': 108,
                     'Putting a cup/glass/bottle somewhere': 109, 'Taking a cup/glass/bottle from somewhere': 110,
                     'Washing a cup/glass/bottle': 111, 'Closing a closet/cabinet': 112,
                     'Opening a closet/cabinet': 113, 'Tidying up a closet/cabinet': 114,
                     'Someone is holding a paper/notebook': 115, 'Putting their paper/notebook somewhere': 116,
                     'Taking paper/notebook from somewhere': 117, 'Holding a dish': 118,
                     'Putting a dish/es somewhere': 119, 'Taking a dish/es from somewhere': 120,
                     'Wash a dish/dishes': 121, 'Lying on a sofa/couch': 122, 'Sitting on sofa/couch': 123,
                     'Lying on the floor': 124, 'Sitting on the floor': 125, 'Throwing something on the floor': 126,
                     'Tidying something on the floor': 127, 'Holding some medicine': 128,
                     'Taking/consuming some medicine': 129, 'Putting groceries somewhere': 130,
                     'Laughing at television': 131, 'Watching television': 132, 'Someone is awakening in bed': 133,
                     'Lying on a bed': 134, 'Sitting in a bed': 135, 'Fixing a vacuum': 136, 'Holding a vacuum': 137,
                     'Taking a vacuum from somewhere': 138, 'Washing their hands': 139, 'Fixing a doorknob': 140,
                     'Grasping onto a doorknob': 141, 'Closing a refrigerator': 142, 'Opening a refrigerator': 143,
                     'Fixing their hair': 144, 'Working on paper/notebook': 145, 'Someone is awakening somewhere': 146,
                     'Someone is cooking something': 147, 'Someone is dressing': 148, 'Someone is laughing': 149,
                     'Someone is running somewhere': 150, 'Someone is going from standing to sitting': 151,
                     'Someone is smiling': 152, 'Someone is sneezing': 153,
                     'Someone is standing up from somewhere': 154, 'Someone is undressing': 155,
                     'Someone is eating something': 156}
    return action_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Charades(Dataset):
    def __init__(self, root, type='train', total_length=12, action_class_list=[]):
        self.root = os.path.expanduser(root)
        self.type = type
        self.total_length = total_length
        self.anno_path = os.path.join(self.root, 'Charades_v1_' + self.type + '.csv')
        self.action_num_class = len(action_class_list)
        self.random_shift = False

        self.total_rows = 7985 if type == 'train' else 1863

        self.transform = self.get_train_transforms()
        try:
            self.label_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

        self.data_len = len(self.label_list)

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        label_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, total=self.total_rows):
                actions = row['actions']
                if actions == '': continue
                vid = row['id']
                path = os.path.join(self.root, 'Charades_v1_rgb', vid)
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.action_num_class), dtype=bool)
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in [a.split(' ') for a in actions.split(';')]]
                for frame in range(num_frames):
                    for ann in actions:
                        if frame/fps > ann[1] and frame/fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                label_list.append([path, num_frames, labels[idx]])
        print("数据集读取完成")
        return label_list, file_list

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        images_names = self.file_list[index]
        label_lists = self.label_list[index]
        indices = self._sample_indices(label_lists[1])
        return self._get(label_lists, images_names, indices)

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    def _get(self, label_lists, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(label_lists[0], image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(os.path.join(label_lists[0], image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        action_label = label_lists[2][indices].any(0).astype(np.float32)
        return process_data, action_label

    def _load_image(self, directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]

    def get_train_transforms(self,):
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                          GroupRandomHorizontalFlip(True),
                          GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                          GroupRandomGrayscale(p=0.2),
                          GroupGaussianBlur(p=0.0),
                          GroupSolarization(p=0.0)])
        common = Compose([Stack(roll=False),
                          ToTorchFormatTensor(div=True),
                          GroupNormalize(input_mean, input_std)])
        transforms = Compose([unique, common])
        return transforms

# ----------------------------------------transforms--------------------------------------------------
class GroupMultiScaleCrop(object):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

    @staticmethod
    def fill_fc_fix_offset(image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 2
        h_step = (image_h - crop_h) // 2

        ret = list()
        ret.append((0, 0))  # left
        ret.append((1 * w_step, 1 * h_step))  # center
        ret.append((2 * w_step, 2 * h_step))  # right

        return ret


class GroupRandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __init__(self, is_sth=False):
        self.is_sth = is_sth

    def __call__(self, img_group, is_sth=False):
        v = random.random()
        if not self.is_sth and v < 0.5:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            return ret
        else:
            return img_group


class GroupRandomColorJitter(object):
    """Randomly ColorJitter the given PIL.Image with a probability
    """

    def __init__(self, p=0.8, brightness=0.4, contrast=0.4,
                 saturation=0.2, hue=0.1):
        self.p = p
        self.worker = T.ColorJitter(brightness=brightness, contrast=contrast,
                                    saturation=saturation, hue=hue)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group


class GroupRandomGrayscale(object):
    """Randomly Grayscale flips the given PIL.Image with a probability
    """

    def __init__(self, p=0.2):
        self.p = p
        self.worker = T.Grayscale(num_output_channels=3)

    def __call__(self, img_group):

        v = random.random()
        if v < self.p:
            ret = [self.worker(img) for img in img_group]

            return ret
        else:
            return img_group

class GroupGaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return [img.filter(ImageFilter.GaussianBlur(sigma))  for img in img_group]
        else:
            return img_group

class GroupSolarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img_group):
        if random.random() < self.p:
            return [ImageOps.solarize(img)  for img in img_group]
        else:
            return img_group

class Stack(object):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                rst = np.concatenate(img_group, axis=2)
                # plt.imshow(rst[:,:,3:6])
                # plt.show()
                return rst

class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """
    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:
            # handle PIL Image
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))
            # put it from HWC to CHW format
            # yikes, this transpose takes 80% of the loading time/CPU
            img = img.transpose(0, 1).transpose(0, 2).contiguous()
        return img.float().div(255) if self.div else img.float()

class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = self.mean * (tensor.size()[0]//len(self.mean))
        std = self.std * (tensor.size()[0]//len(self.std))
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)

        if len(tensor.size()) == 3:
            # for 3-D tensor (T*C, H, W)
            tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
        elif len(tensor.size()) == 4:
            # for 4-D tensor (C, T, H, W)
            tensor.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
        return tensor
# -*- coding: utf-8 -*-
# Author: Xiangde Luo
# Date:   16 Dec. 2021
# Implementation for Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer.
# # Reference:
#   @article{luo2021ctbct,
#   title={Semi-Supervised Medical Image Segmentation via Cross Teaching between CNN and Transformer},
#   author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
#   journal={arXiv preprint arXiv:2112.04894},
#   year={2021}}
#   In the original paper, we don't use the validation set to select checkpoints and use the last iteration to inference for all methods.
#   In addition, we combine the validation set and test set to report the results.
#   We found that the random data split has some bias (the validation set is very tough and the test set is very easy).
#   Actually, this setting is also a fair comparison.
#   download pre-trained model to "code/pretrained_ckpt" folder, link:https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY

import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights

from config import get_config
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from networks.vision_transformer import SwinUnet as ViT_seg
from utils import losses, metrics, ramps
from val_2D import test_single_volume
from networks.RadImageNet_pretrained_models import get_compiled_RadImageNet_model
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC/Cross_Teaching_Between_CNN_Transformer', help='experiment_name')
parser.add_argument('--model1_weights', type=str,
                    default='../model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/best_overall_models/unet_best_model1.pth', help='path to model1 .pth file')
parser.add_argument('--model2_weights', type=str,
                    default='../model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/best_overall_models/unet_best_model2.pth', help='path to model2 .pth file')
parser.add_argument('--model', type=str, choices=['simple', 'resnet18', 'resnet34', 'resnet50'],
                    default='resnet34', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.005,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument(
    '--cfg', type=str, default="../code/configs/swin_tiny_patch4_window7_224_lite.yaml", help='path to config file', )
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=7,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.01, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()
config = get_config(args)


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(128, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type="unet", in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    def create_selector_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.selector_model, in_chns=1,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    # selector = get_compiled_RadImageNet_model('IRV2', args.patch_size[0], base_lr)
    # selector = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    num_channels = 1
    val_img_size = 256
    if args.model == "resnet18":
        selector = resnet18(weights=ResNet18_Weights.DEFAULT)
        selector.fc = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 2)
            )
        num_channels = 3
    elif args.model == "resnet34":
        selector = resnet34(weights=ResNet34_Weights.DEFAULT)
        selector.fc = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.BatchNorm1d(256),
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 2)
            )
        num_channels = 3
    elif args.model == "resnet50":
        selector = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        selector.heads = nn.Sequential(
              nn.Linear(768, 512),
              nn.ReLU(),
              nn.Linear(512, 256),
              nn.ReLU(),
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 2)
            )
        num_channels = 3
        val_img_size = 224
    elif args.model == "simple":
        selector = SimpleNet()
    model1 = create_model()
    model2 = ViT_seg(config, img_size=args.patch_size,
                     num_classes=args.num_classes)# .cuda()
    model2.load_from(config)

    model1.load_state_dict(torch.load(args.model1_weights))
    model2.load_state_dict(torch.load(args.model2_weights))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        RandomGenerator(args.patch_size)
    ]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)  # num_workers was 4

    model1.eval()
    model2.eval()
    selector.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=0)  # num_workers was 1

    optimizer = optim.SGD(selector.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss(reduction='none')
    selector_ce_loss = CrossEntropyLoss()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']

            outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2 = model2(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            loss1 = ce_loss(outputs1.detach(), label_batch.long()).mean(dim=(1, 2))

            loss2 = ce_loss(outputs2.detach(), label_batch.long()).mean(dim=(1, 2))

            consistency_weight = get_current_consistency_weight(
                iter_num // 150)

            selector_output = selector(volume_batch.repeat(1,num_channels,1,1))
            selector_soft = torch.softmax(selector_output, dim=1)

            # Stack the loss vectors to create a matrix of shape (N, 2)
            loss_matrix = torch.stack([loss1.detach(), loss2.detach()], dim=1)  # Shape: (N, 2)

            # Find the index of the minimal value in each row
            _, min_indices = loss_matrix.min(dim=1)  # min_indices will be of shape (N,)

            # Create a tensor of shape (N, 2) with zeros
            one_hot_tensor = torch.zeros(loss1.shape[0], 2)

            # Set the position of the minimal value to 1
            one_hot_tensor.scatter_(1, min_indices.unsqueeze(1), 1)

            # norm = (sum(selector_soft[:, 0] - selector_soft[:, 1]).abs()**(-1) + (sum(selector_soft[:,0]) - sum(selector_soft[:,1])).abs())
            norm = selector_soft[:, 0].var()**(-1) + selector_soft[:, 1].var()**(-1)
            loss = selector_ce_loss(selector_soft, one_hot_tensor.detach()) + 0.00001 * norm

            print(f"chose model1 {len(selector_soft)-sum(torch.argmax(torch.softmax(selector_soft, dim=1), dim=1).squeeze(0))}/{len(selector_soft)}")
            print(f"model1 was better {len(selector_soft)-sum(torch.argmax(one_hot_tensor, dim=1).squeeze(0))}/{len(selector_soft)}")

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/selector_loss',
                              loss, iter_num)
            logging.info('iteration %d : loss : %f' % (
                iter_num, loss.item()))

            if iter_num > 0 and iter_num % 100 == 0:
                model1.eval()
                metric_list1 = []
                model2.eval()
                metric_list2 = []
                selector.eval()
                selector_pred = []
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list1.append(np.array(metric_i))

                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model2, classes=num_classes,
                        patch_size=args.patch_size)
                    metric_list2.append(np.array(metric_i))

                    image = sampled_batch["image"].squeeze(0).cpu().detach().numpy()
                    selector_output = []
                    for ind in range(image.shape[0]):
                        slice = image[ind, :, :]
                        x, y = slice.shape[0], slice.shape[1]
                        slice = zoom(slice, (val_img_size / x, val_img_size / y), order=0)
                        input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
                        with torch.no_grad():
                            initial_output = selector(input.repeat(1, num_channels, 1, 1))
                            selector_output.append(initial_output)
                    selector_output_sum = sum(selector_output)
                    selector_pred_i = torch.argmax(torch.softmax(selector_output_sum, dim=1), dim=1).squeeze(0)
                    selector_pred.append(selector_pred_i)

                mean_metric1 = sum(metric_list1) / len(db_val)

                performance1 = np.mean(mean_metric1, axis=0)[0]

                mean_hd951 = np.mean(mean_metric1, axis=0)[1]

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))

                mean_metric2 = sum(metric_list2) / len(db_val)

                performance2 = np.mean(mean_metric2, axis=0)[0]

                mean_hd952 = np.mean(mean_metric2, axis=0)[1]

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' % (iter_num, performance2, mean_hd952))


                # for i_batch, sampled_batch in enumerate(valloader):
                #     image = sampled_batch["image"].squeeze(0).cpu().detach().numpy()
                #     selector_output = np.zeros(image.shape[0])
                #     for ind in range(image.shape[0]):
                #         slice = image[ind, :, :]
                #         x, y = slice.shape[0], slice.shape[1]
                #         slice = zoom(slice, (256 / x, 256 / y), order=0)
                #         input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float()
                #         with torch.no_grad():
                #             initial_output = selector(input.repeat(1,3,1,1))
                #             selector_output[ind] = torch.argmax(torch.softmax(initial_output, dim=1), dim=1).squeeze(0)
                selector.train()

                selector_metric_list = []
                num_chose_1 = 0
                for i in range(len(metric_list1)):
                    if selector_pred[i] == 0:
                        selector_metric_list.append(metric_list1[i])
                        num_chose_1 += 1
                    else:
                        selector_metric_list.append(metric_list2[i])
                selector_mean_metric = sum(selector_metric_list) / len(db_val)

                selector_performance = np.mean(selector_mean_metric, axis=0)[0]

                selector_mean_hd95 = np.mean(selector_mean_metric, axis=0)[1]

                print(f"VAL : chose model 1 {num_chose_1}/{len(selector_pred)}")

                logging.info(
                    'iteration %d : selector_mean_dice : %f selector_mean_hd95 : %f' % (iter_num, selector_performance, selector_mean_hd95))

                if selector_performance > best_performance:
                    best_performance = selector_performance
                    save_mode_path = os.path.join(snapshot_path,
                                                  'selector_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             'selector_best.pth')
                    torch.save(selector.state_dict(), save_mode_path)
                    torch.save(selector.state_dict(), save_best)

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    snapshot_path = "../selector/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)

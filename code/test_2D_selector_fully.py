import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import zoom
from tqdm import tqdm
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights

# from networks.efficientunet import UNet
from networks.net_factory import net_factory

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/ACDC', help='Name of Experiment')
parser.add_argument('--model1_weights', type=str,
                    default='../model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/best_overall_models/unet_best_model1.pth', help='path to model1 .pth file')
parser.add_argument('--model2_weights', type=str,
                    default='../model/ACDC/Cross_Teaching_Between_CNN_Transformer_7/best_overall_models/unet_best_model2.pth', help='path to model2 .pth file')
parser.add_argument('--selector_weights', type=str,
                    default='../selector/ACDC/Cross_Teaching_Between_CNN_Transformer_7/resnet34_from_server/resnet34/selector_best.pth', help='path to model .pth file')
parser.add_argument('--exp', type=str,
                    default='ACDC/Fully_Supervised', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--selector_model', choices=['resnet18', 'resnet34'], type=str,
                    default='resnet34', help='model_name')
parser.add_argument('--num_classes', type=int,  default=4,
                    help='output channel of network')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[224, 224],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
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
parser.add_argument('--loss', type=str, default='wce', choices=['ce_and_dice', 'nll', 'wce'],
                    help='loss function to use for the labeled data training')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice, hd95, asd


def test_single_volume(case, model1, model2, selector, test_save_path, FLAGS):
    h5f = h5py.File(FLAGS.root_path + "/data/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction  = np.zeros_like(label)
    prediction1 = np.zeros_like(label)
    prediction2 = np.zeros_like(label)
    num_selected_model1 = 0
    total_num = 0
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        slice = zoom(slice, (FLAGS.patch_size[0] / x, FLAGS.patch_size[1] / y), order=0)
        input = torch.from_numpy(slice).unsqueeze(
            0).unsqueeze(0).float()#.cuda()
        with torch.no_grad():
            initial_output = selector(input.repeat(1, 3, 1, 1))
            selector_pred = torch.argmax(torch.softmax(initial_output, dim=1), dim=1)
            total_num += 1
            out_main1 = model1(input)
            out_main2 = model2(input)

            out1 = torch.argmax(torch.softmax(
                out_main1, dim=1), dim=1).squeeze(0)
            out1 = out1.cpu().detach().numpy()
            pred1 = zoom(out1, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction1[ind] = pred1

            out2 = torch.argmax(torch.softmax(
                out_main2, dim=1), dim=1).squeeze(0)
            out2 = out2.cpu().detach().numpy()
            pred2 = zoom(out2, (x / FLAGS.patch_size[0], y / FLAGS.patch_size[1]), order=0)
            prediction2[ind] = pred2

            if selector_pred == 0:
                num_selected_model1 += 1
                prediction[ind] = pred1
            else:
                prediction[ind] = pred2

    first_metric1 = calculate_metric_percase(prediction1 == 1, label == 1)
    second_metric1 = calculate_metric_percase(prediction1 == 2, label == 2)
    third_metric1 = calculate_metric_percase(prediction1 == 3, label == 3)

    first_metric2 = calculate_metric_percase(prediction2 == 1, label == 1)
    second_metric2 = calculate_metric_percase(prediction2 == 2, label == 2)
    third_metric2 = calculate_metric_percase(prediction2 == 3, label == 3)

    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")
    return first_metric, second_metric, third_metric, num_selected_model1, total_num


def Inference(FLAGS):
    with open(FLAGS.root_path + '/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])

    test_save_path = os.path.splitext(FLAGS.selector_weights)[0] + "_predictions/selector/"
    save_mode_path1 = FLAGS.model1_weights
    save_mode_path2 = FLAGS.model2_weights

    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)

    model1 = net_factory(net_type=FLAGS.model, in_chns=1,
                      class_num=FLAGS.num_classes)
    FLAGS2 = FLAGS
    FLAGS2.model = "ViT_Seg"
    model2 = net_factory(net_type=FLAGS2.model, in_chns=1,
                      class_num=FLAGS2.num_classes)

    model1.load_state_dict(torch.load(save_mode_path1))
    print("init weight for model1 from {}".format(save_mode_path1))
    model1.eval()

    model2.load_state_dict(torch.load(save_mode_path2))
    print("init weight for model2 from {}".format(save_mode_path2))
    model2.eval()

    if FLAGS.selector_model == "resnet18":
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
    elif FLAGS.selector_model == "resnet34":
        selector = resnet34(weights=ResNet34_Weights.DEFAULT)
        selector.fc = nn.Sequential(
              nn.Linear(512, 256),
              nn.ReLU(),
              # nn.BatchNorm1d(256),
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 2)
            )
    selector.load_state_dict(torch.load(FLAGS.selector_weights))
    print("init weight for selector from {}".format(FLAGS.selector_weights))
    selector.eval()

    first_total = 0.0
    second_total = 0.0
    third_total = 0.0
    num_selected_model1 = 0
    total_num = 0
    for case in tqdm(image_list):
        first_metric, second_metric, third_metric, num_selected_model1_i, total_num_i = test_single_volume(
            case, model1, model2, selector, test_save_path, FLAGS)
        first_total += np.asarray(first_metric)
        second_total += np.asarray(second_metric)
        third_total += np.asarray(third_metric)
        num_selected_model1 += num_selected_model1_i
        total_num += total_num_i
    print(f"Selected model1: {num_selected_model1}/{total_num}")
    avg_metric = [first_total / len(image_list), second_total /
                  len(image_list), third_total / len(image_list)]
    return avg_metric


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    metric = Inference(FLAGS)
    print(metric)
    print(f"[dice, hd95, asd] = {(metric[0]+metric[1]+metric[2])/3}")

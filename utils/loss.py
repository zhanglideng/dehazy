import torch
from utils.ms_ssim import *
import math
import torch.nn.functional


def l2_loss(input_image, output_image):
    l2_loss_fn = torch.nn.MSELoss(reduction='mean').cuda()
    return l2_loss_fn(input_image, output_image) * 100


def ssim_loss(input_image, output_image, channel=3):
    losser = MS_SSIM(max_val=1, channel=channel).cuda()
    # losser = MS_SSIM(data_range=1.).cuda()
    return (1 - losser(input_image, output_image)) * 100


'''
def color_loss(image, label, len_reg=0):
    vec1 = tf.reshape(image, [-1, 3]) 想要得到一个第二维度是3的向量，不管第一维度是几。
    vec2 = tf.reshape(label, [-1, 3])
    clip_value = 0.999999
    norm_vec1 = tf.nn.l2_normalize(vec1, 1) l2范化
    norm_vec2 = tf.nn.l2_normalize(vec2, 1)
    dot = tf.reduce_sum(norm_vec1*norm_vec2, 1) 维度求和
    dot = tf.clip_by_value(dot, -clip_value, clip_value) 限制在一个范围内
    angle = tf.acos(dot) * (180/math.pi) 根据值反求角度
    return tf.reduce_mean(angle) 计算角度的均值
'''


def color_loss(input_image, output_image):
    vec1 = input_image.view([-1, 3])
    vec2 = output_image.view([-1, 3])
    clip_value = 0.999999
    norm_vec1 = torch.nn.functional.normalize(vec1)
    norm_vec2 = torch.nn.functional.normalize(vec2)
    dot = norm_vec1 * norm_vec2
    dot = dot.mean(dim=1)
    dot = torch.clamp(dot, -clip_value, clip_value)
    angle = torch.acos(dot) * (180 / math.pi)
    return angle.mean()


def loss_At_function(image, weight):
    gt, A_haze, t_haze, J, A, t = image
    loss_train = [l2_loss(gt, J),
                  ssim_loss(gt, J),
                  l2_loss(A_haze, A),
                  l2_loss(t_haze, t),
                  ssim_loss(t_haze, t, channel=1)]
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
    return loss_sum, loss_train


def loss_ed_function(image, weight):
    output, gt, dehaze, gt_scene, hazy_scene = image
    loss_train = [l2_loss(gt, output),
                  ssim_loss(gt, output),
                  l2_loss(gt, dehaze),
                  ssim_loss(gt, dehaze),
                  l2_loss(gt_scene, hazy_scene)]
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
    return loss_sum, loss_train


def loss_recon_function(image, weight):
    haze, I = image
    loss_train = [l2_loss(haze, I),
                  ssim_loss(haze, I)]
    loss_sum = 0
    for i in range(len(loss_train)):
        loss_sum = loss_sum + loss_train[i] * weight[i]
        loss_train[i] = loss_train[i].item()
    return loss_sum, loss_train

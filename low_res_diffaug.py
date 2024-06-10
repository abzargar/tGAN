import torch
import torch.nn.functional as F
import random

def DiffAugment(masks,imgs,p=0.5):
    if random.random()<p:
        orders=[i for i in range(len(AUGMENT_FNS))]
        random.shuffle(orders)
        for i in orders:
            imgs,masks = AUGMENT_FNS[i](imgs,masks)
    return masks.contiguous(),imgs.contiguous()


def rand_brightness(imgs,masks):
    imgs = imgs + (torch.rand(imgs.size(0), 1, 1, 1, dtype=imgs.dtype, device=imgs.device) - 0.5)
    return imgs,masks


def rand_saturation(imgs,masks):
    imgs_mean = imgs.mean(dim=1, keepdim=True)
    imgs = (imgs - imgs_mean) * (torch.rand(imgs.size(0), 1, 1, 1, dtype=imgs.dtype, device=imgs.device) * 2) + imgs_mean
    return imgs,masks


def rand_contrast(imgs,masks):
    imgs_mean = imgs.mean(dim=[1, 2, 3], keepdim=True)
    imgs = (imgs - imgs_mean) * (torch.rand(imgs.size(0), 1, 1, 1, dtype=imgs.dtype, device=imgs.device) + 0.5) + imgs_mean
    return imgs,masks


def rand_translation(imgs,masks, ratio=0.125):
    shift_x, shift_y = int(imgs.size(2) * ratio + 0.5), int(imgs.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[imgs.size(0), 1, 1], device=imgs.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[imgs.size(0), 1, 1], device=imgs.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(imgs.size(0), dtype=torch.long, device=imgs.device),
        torch.arange(imgs.size(2), dtype=torch.long, device=imgs.device),
        torch.arange(imgs.size(3), dtype=torch.long, device=imgs.device),indexing='ij'
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, imgs.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, imgs.size(3) + 1)
    imgs_pad = F.pad(imgs, [1, 1, 1, 1, 0, 0, 0, 0])
    masks_pad = F.pad(masks, [1, 1, 1, 1, 0, 0, 0, 0])
    imgs = imgs_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    masks = masks_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2).contiguous()
    return imgs,masks


def rand_cutout(imgs,masks, ratio=0.2):
    cutout_size = int(imgs.size(2) * ratio + 0.5), int(imgs.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, imgs.size(2) + (1 - cutout_size[0] % 2), size=[imgs.size(0), 1, 1], device=imgs.device)
    offset_y = torch.randint(0, imgs.size(3) + (1 - cutout_size[1] % 2), size=[imgs.size(0), 1, 1], device=imgs.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(imgs.size(0), dtype=torch.long, device=imgs.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=imgs.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=imgs.device),indexing='ij'
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=imgs.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=imgs.size(3) - 1)
    mask = torch.ones(imgs.size(0), imgs.size(2), imgs.size(3), dtype=imgs.dtype, device=imgs.device)
    mask[grid_batch, grid_x, grid_y] = 0
    imgs = imgs * mask.unsqueeze(1)
    masks = masks * mask.unsqueeze(1)
    return imgs,masks

AUGMENT_FNS =  [rand_brightness, rand_saturation, rand_contrast,rand_translation,rand_cutout]


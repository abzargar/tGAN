# # Import the necessary libraries and modules
import os
import argparse
from low_res_models import Vid2VidStyleGenerator, NLayerDiscriminator,ComplexFlowNet
from utils import set_requires_grad, image_noise, mixed_list, noise_list,apply_flow_to_output,make_video
from low_res_data import BasicDataset
import itertools
import torch.nn as nn
from loss import VGGLoss
import torch.utils.data as data
import low_res_transforms as transforms
import torch
import random
import logging
from tqdm import tqdm
import numpy as np
from low_res_diffaug import DiffAugment

# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# The main training function
def train(args, image_size=[256, 384], image_means=[0.5], image_stds=[0.5]):
    # Set up the logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',
                        format='%(asctime)s - %(message)s', level=logging.INFO)

    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (
    image_size[0], image_size[1], args.lr, args.batch_size))

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],
                                   p=0.5),
            transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
            transforms.RandomApply([transforms.CLAHE()], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
            transforms.RandomApply([transforms.RandomCrop()], p=0.5),
        ])], p=args.p_vanilla),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Define the datasets for training and validation
    train_data = BasicDataset(os.path.join(args.train_set_dir, 'seg_maps'),
                              os.path.join(args.train_set_dir, 'images'), transforms=train_transforms,n_frames_total=args.n_frames_total)

    # Define the dataloaders
    train_iterator = data.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, num_workers=2,
                                     pin_memory=True)

    # Define the models
    Gen = Vid2VidStyleGenerator(style_latent_dim=128)
    FlowNet=ComplexFlowNet()
    D = NLayerDiscriminator()

    # Define the optimizers
    optimizer_G = torch.optim.RMSprop(itertools.chain(Gen.parameters()), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = args.max_epoch * STEPS_PER_EPOCH * args.max_sequence_length

    # Define the learning rate schedulers
    MAX_LRS_G = [p['lr'] for p in optimizer_G.param_groups]
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=MAX_LRS_G, total_steps=TOTAL_STEPS)

    MAX_LRS_D = [p['lr'] for p in optimizer_D.param_groups]
    scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=MAX_LRS_D, total_steps=TOTAL_STEPS)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Define the loss functions
    d_criterion = nn.MSELoss()
    Vgg_criterion = VGGLoss()
    L1_criterion = nn.L1Loss()
    temporal_criterion = nn.L1Loss()
    flow_criterion = nn.L1Loss()

    # Move everything to the device
    Gen = nn.DataParallel(Gen.to(device))
    D = nn.DataParallel(D.to(device))
    FlowNet = nn.DataParallel(FlowNet.to(device))
    d_criterion = d_criterion.to(device)
    Vgg_criterion = Vgg_criterion.to(device)
    L1_criterion = L1_criterion.to(device)
    temporal_criterion = temporal_criterion.to(device)
    flow_criterion = flow_criterion.to(device)

    # Training loop
    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    tG = 2
    input_nc = 1
    output_nc = 1
    n_frames_load = args.initial_sequence_length  # number of total frames loaded into GPU at a time for each batch
    n_frames_load = min(n_frames_load, args.n_frames_total - tG + 1)

    for epoch in range(args.max_epoch):
        Gen.train()
        D.train()
        FlowNet.train()
        if epoch>0 and epoch % args.num_epochs_temporal_step == 0:
            n_frames_load += 1
        if n_frames_load > args.max_sequence_length:
            n_frames_load = args.max_sequence_length

        t_len = n_frames_load + tG - 1
        g_loss_total, d_loss_total = 0, 0
        for idx, data_list in enumerate(tqdm(train_iterator)):

            for i in range(0, args.n_frames_total - n_frames_load):
                _, _, h, w = data_list['B'].size()
                input_A = (data_list['A'][:, i * input_nc:(i + t_len) * input_nc, ...]).view(-1, t_len, input_nc, h, w)
                input_B = (data_list['B'][:, i * output_nc:(i + t_len) * output_nc, ...]).view(-1, t_len, output_nc, h,
                                                                                               w)
                bg_B = (data_list['B'][:, args.n_frames_total * output_nc, ...]).view(-1, 1, output_nc, h, w)
                bg_A = (data_list['A'][:, args.n_frames_total * output_nc, ...]).view(-1, 1, output_nc, h, w)
                composite_mask = (torch.max(bg_A) - bg_A) / torch.max(bg_A)
                bg_B[composite_mask == 0] = torch.min(bg_B)
                bg_ref = bg_B.to(device=device, dtype=torch.float32)
                input_A = input_A.to(device=device, dtype=torch.float32)
                input_B = input_B.to(device=device, dtype=torch.float32)

                fake_B_pyr = input_B[:, :(tG - 1), ...].clone()
                for t in range(n_frames_load):
                    real_A = 0.4 * input_A[:, t:t + tG, ...] / torch.max(input_A[:, t:t + tG, ...])
                    real_B = input_B[:, t:t + tG, ...]
                    real_As_selected = real_A.view(input_B.shape[0], -1, h, w)
                    bg_ref = bg_ref.view(input_B.shape[0], -1, h, w)
                    if t < tG:
                        fake_B_prev = fake_B_pyr[:, t: t + tG - 1, ...]
                    else:
                        fake_B_prev = fake_B_pyr[:, 1: 1 + tG - 1, ...]

                    fake_B_prev = fake_B_prev.detach()
                    fake_B_prev_selected = fake_B_prev.clone().view(input_B.shape[0], -1, h, w)

                    if t == 0:
                        composite_mask = (real_A[:, -2, ...] - real_A[:, -2, ...]) / torch.max(real_A)
                        fake_B_prev_selected[composite_mask == 0] = torch.min(fake_B_prev_selected)

                    if random.random() < 0.9:
                        style = mixed_list(fake_B_prev_selected.shape[0], 5, Gen.module.latent_dim, device=device)
                    else:
                        style = noise_list(fake_B_prev_selected.shape[0], 5, Gen.module.latent_dim, device=device)

                    im_noise = image_noise(fake_B_prev_selected.shape[0], image_size, device=device)
                    fake_B = Gen(real_As_selected, fake_B_prev_selected, bg_ref, style, im_noise)
                    fake_B_pyr = torch.cat([fake_B_prev, fake_B.unsqueeze(1)], dim=1)

                    if t == 0:
                        output_B = fake_B.unsqueeze(1).clone()
                    else:
                        output_B = torch.cat([output_B, fake_B.unsqueeze(1)], dim=1)

                    fake_B_current = fake_B_pyr[:, -1, ...]

                    real_B_prev, real_B_current = real_B[:, -2, ...], real_B[:, -1, ...]

                    real_B_current = real_B_current.contiguous().view(-1, 1, h, w)
                    real_B_prev = real_B_prev.contiguous().view(-1, 1, h, w)
                    fake_B_current = fake_B_current.contiguous().view(-1, 1, h, w)

                    set_requires_grad(D, False)
                    set_requires_grad(Gen, True)
                    set_requires_grad(FlowNet, True)
                    optimizer_G.zero_grad(set_to_none=True)

                    # Calculate the "real flow" between real previous frame and real current frame
                    real_flow = FlowNet(real_B_prev, real_B_current)
                    estimated_real_B_current = apply_flow_to_output(real_B_prev, real_flow)
                    loss_temporal = temporal_criterion(real_B_current, estimated_real_B_current)

                    # Calculate the "fake flow" between real previous frame and generated current frame
                    fake_flow = FlowNet(real_B_prev, fake_B_current)

                    # Calculate the optical flow loss
                    flow_loss = flow_criterion(real_flow, fake_flow)

                    Vgg_img_loss = Vgg_criterion(fake_B, real_B_current)
                    L1_img_loss = L1_criterion(fake_B, real_B_current)

                    real_AB = torch.cat((real_A.contiguous().view(-1, 2, h, w), real_B.contiguous().view(-1, 2, h, w)),dim=1)
                    aug_real_A, aug_fake_B = DiffAugment(real_A.contiguous().view(-1, 2, h, w), fake_B_pyr.contiguous().view(-1, 2, h, w),p=args.p_diff)
                    aug_fake_AB = torch.cat((aug_real_A, aug_fake_B), dim=1)
                    valid = torch.full((real_AB.shape[0], 1, 30, 46), 1.0, dtype=real_AB.dtype, device=device)
                    fake = torch.full((real_AB.shape[0], 1, 30, 46), 0.0, dtype=real_AB.dtype, device=device)

                    d_img_loss = d_criterion(D(aug_fake_AB), valid)

                    g_loss = d_img_loss + 100 * Vgg_img_loss + 0 * L1_img_loss + 100 * flow_loss + 50 * loss_temporal
                    g_loss_total += g_loss
                    grad_scaler.scale(g_loss).backward()  # Scale the loss, and then backward pass
                    grad_scaler.step(optimizer_G)  # Update optimizer with scaled gradients
                    grad_scaler.update()  # Update the scale for next iteration
                    scheduler_G.step()

                    set_requires_grad(D, True)
                    optimizer_D.zero_grad(set_to_none=True)
                    aug_real_A, aug_real_B = DiffAugment(real_A.contiguous().view(-1, 2, h, w), real_B.contiguous().view(-1, 2, h, w), p=args.p_diff)
                    aug_real_AB = torch.cat((aug_real_A, aug_real_B), dim=1)
                    real_img_loss = d_criterion(D(aug_real_AB), valid)
                    aug_real_A, aug_fake_B = DiffAugment(real_A.contiguous().view(-1, 2, h, w),fake_B_pyr.contiguous().view(-1, 2, h, w), p=args.p_diff)
                    aug_fake_AB = torch.cat((aug_real_A, aug_fake_B), dim=1)
                    fake_img_loss = d_criterion(D(aug_fake_AB.detach()), fake)
                    d_img_loss = (real_img_loss + fake_img_loss) / 2
                    d_loss_total += d_img_loss

                    grad_scaler.scale(d_img_loss).backward()
                    grad_scaler.step(optimizer_D)
                    grad_scaler.update()
                    scheduler_D.step()

        # make_video(input_B, os.path.join(args.output_dir, f'real_B_epoch_idx_{idx}.mp4'))
        # make_video(input_A, os.path.join(args.output_dir, f'real_A_epoch_idx_{idx}.mp4'))
        # make_video(output_B, os.path.join(args.output_dir, f'fake_B_epoch_idx_{idx}.mp4'))

        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.max_epoch, d_loss_total, g_loss_total)
        )
        logging.info("[Epoch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, args.max_epoch, d_loss_total, g_loss_total))

        if epoch % args.save_ckpt_interval == 0:
            torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'low_res_Gen.pth'))
            torch.save(D.state_dict(), os.path.join(args.output_dir, 'low_res_D.pth'))
            torch.save(FlowNet.state_dict(), os.path.join(args.output_dir, 'low_res_FlowNet.pth'))

# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir", required=True, type=str, help="path for the train set")
    ap.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    ap.add_argument("--max_epoch", default=2500, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=2, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and checkpoints")
    ap.add_argument("--p_vanilla", default=0.2, type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_diff", default=0.2, type=float,help="probability value of diff augmentation, a value between 0 and 1")
    ap.add_argument("--initial_sequence_length", default=1, type=int, help="initial training clip length")
    ap.add_argument("--max_sequence_length", default=10, type=int,help="max training clip length")
    ap.add_argument("--n_frames_total", default=20, type=int, help="total number of frames to load from each clip")
    ap.add_argument("--num_epochs_temporal_step", default=50, type=int,help="how often to increment the number of training frames in each clip")
    ap.add_argument('--save_ckpt_interval', type=int, default=10, help="Specify how often (in epochs) to save the best checkpoint")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)

# # Import the necessary libraries and modules
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,7"
import argparse
from models_14 import StyleUnetGenerator,NLayerDiscriminator,ComplexFlowNet
from utils import set_requires_grad,image_noise,mixed_list,noise_list
from data_3 import BasicDataset
import itertools
import torch.nn as nn
import cv2
from loss import VGGLoss
import torch.utils.data as data
import torch.nn.functional as F
import transforms_2 as transforms
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
from vid_diffaug import DiffAugment


# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

if not os.path.exists('frames_real_B_21'):
    os.mkdir('frames_real_B_21')
if not os.path.exists('frames_fake_B_21'):
    os.mkdir('frames_fake_B_21')
if not os.path.exists('frames_real_A_21'):
    os.mkdir('frames_real_A_21')

if not os.path.exists('frames_real_B_21_test'):
    os.mkdir('frames_real_B_21_test')
if not os.path.exists('frames_fake_B_21_test'):
    os.mkdir('frames_fake_B_21_test')
if not os.path.exists('frames_real_A_21_test'):
    os.mkdir('frames_real_A_21_test')




def apply_flow_to_output(image, flow):
    b, _, h, w = image.size()
    x = torch.linspace(0, w - 1, w, device=image.device)
    y = torch.linspace(0, h - 1, h, device=image.device)
    # y, x = torch.meshgrid(y, x)
    x, y = torch.meshgrid(x, y, indexing='xy')

    x = x.expand([b, 1, -1, -1])
    y = y.expand([b, 1, -1, -1])
    grid = torch.cat((x, y), 1).float()

    # Initialize the div_tensor with a compatible shape
    div_tensor = torch.tensor([[[[w - 1, h - 1]]]], dtype=flow.dtype, device=flow.device)

    # Reshape to have same dimensions as `flow` while having 1 in all dimensions where the size doesn't match
    div_tensor = div_tensor.view(1, 2, 1, 1)

    # Now expand dimensions
    div_tensor = div_tensor.expand(flow.size(0), flow.size(1), flow.size(2), flow.size(3))

    # print("div_tensor shape after expand:", div_tensor.shape)

    # Then divide
    flow = flow / div_tensor

    new_grid = grid + flow
    remapped_image = F.grid_sample(image, new_grid.permute(0, 2, 3, 1), mode='bilinear', padding_mode='reflection',
                                   align_corners=False)
    return remapped_image


def compute_fake_B_prev(real_B_prev, fake_B_last, fake_B):
    fake_B_prev = real_B_prev[:, 0:1] if fake_B_last is None else fake_B_last[0][:, -1:]
    if fake_B.size()[1] > 1:
        fake_B_prev = torch.cat([fake_B_prev, fake_B[:, :-1].detach()], dim=1)
    return fake_B_prev
import numpy as np

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / ((max_val - min_val)+1e-6)



def make_video(input,video_name, fps=1):
    bs, n_f, ch, h, w = input.size()
    input = input[0, :, :, :].view(-1, ch, h, w)
    # h, w = 480, 640
    size = (w, h)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=True)

    if not out.isOpened():
        print("Error: Video Writer not initialized.")
        return

    # print("Backend:", out.getBackendName())  # For debugging

    for f_idx in range(n_f):
        frame = input[f_idx, :, :]
        frame = normalize_image(frame.detach().cpu().numpy()) * 255
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # single_frame = np.random.randint(0, 256, (h, w, 3)).astype('uint8')
        out.write(frame)

    out.release()



# The main training function
def train(args,image_size = [256,384],image_means = [0.5],image_stds= [0.5]):
    # Set up the logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info('>>>> image size=(%d,%d) , learning rate=%f , batch size=%d' % (image_size[0], image_size[1],args.lr,args.train_batch_size))

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transforms=transforms.Compose([
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
        ])], p=0.5),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    dev_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means,std=image_stds)
    ])

    # Define the datasets for training and validation
    train_data = BasicDataset(os.path.join(args.train_set_dir, 'train_mask_images'),
                                          os.path.join(args.train_set_dir, 'train_images'),transforms=train_transforms,isTrain=True)
    test_data = BasicDataset(os.path.join(args.train_set_dir, 'test_mask_images'),
                                          os.path.join(args.train_set_dir, 'test_images'),transforms=dev_transforms,isTrain=True)

    # Define the dataloaders
    train_iterator = data.DataLoader(train_data,shuffle = True,batch_size = args.train_batch_size,num_workers=2,pin_memory=True)
    test_iterator = data.DataLoader(test_data,batch_size = args.test_batch_size,num_workers=2 ,pin_memory=True)

    # Define the models
    Gen = StyleUnetGenerator(style_latent_dim = 128)
    Gen_dev = StyleUnetGenerator(style_latent_dim=128)
    FlowNet = ComplexFlowNet()


    # Seg = Segmentation(n_channels=1, n_classes=2, bilinear=True)
    D = NLayerDiscriminator()

    # Define the optimizers
    optimizer_G = torch.optim.RMSprop(itertools.chain(Gen.parameters()), lr=args.lr,weight_decay=1e-8, momentum=0.9)
    optimizer_D = torch.optim.RMSprop(D.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)


    STEPS_PER_EPOCH = len(train_iterator)
    TOTAL_STEPS = args.max_epoch * STEPS_PER_EPOCH*70

    # Define the learning rate schedulers
    MAX_LRS_G = [p['lr'] for p in optimizer_G.param_groups]
    scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer_G, max_lr=MAX_LRS_G, total_steps=TOTAL_STEPS)

    MAX_LRS_D = [p['lr'] for p in optimizer_D.param_groups]
    scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer_D, max_lr=MAX_LRS_D, total_steps=TOTAL_STEPS)

    grad_scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Define the loss functions
    d_criterion = nn.MSELoss()
    Vgg_criterion = VGGLoss()
    L1_criterion=nn.L1Loss()
    temporal_criterion = nn.L1Loss()
    flow_criterion = nn.L1Loss()
    # Move everything to the device
    Gen = nn.DataParallel(Gen.to(device))
    Gen_dev = nn.DataParallel(Gen_dev.to(device))
    # Seg = nn.DataParallel(Seg.to(device))
    D = nn.DataParallel(D.to(device))
    FlowNet = nn.DataParallel(FlowNet.to(device))
    Gen.load_state_dict(torch.load(os.path.join('/home/azargari/cell_vid2vid/train_outputs_23/', 'Gen.pth')))
    D.load_state_dict(torch.load(os.path.join('/home/azargari/cell_vid2vid/train_outputs_23/', 'D.pth')))
    FlowNet.load_state_dict(torch.load(os.path.join('/home/azargari/cell_vid2vid/train_outputs_23/', 'FlowNet.pth')))
    d_criterion=d_criterion.to(device)
    Vgg_criterion=Vgg_criterion.to(device)
    L1_criterion = L1_criterion.to(device)
    temporal_criterion=temporal_criterion.to(device)
    flow_criterion=flow_criterion.to(device)
    # Training loop

    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    # t_scales=2
    n_gpus=1
    p_diff=0
    tG=2
    n_frames_bp=1
    input_nc=1
    output_nc=1
    n_frames_load = 0  # number of total frames loaded into GPU at a time for each batch
    n_frames_total = 13
    n_frames_load = min(n_frames_load, n_frames_total - tG + 1)
    g_loss_min=float('inf')

    for epoch in range(args.max_epoch):
        Gen.train()
        D.train()
        FlowNet.train()
        if epoch%50==0:
            n_frames_load+=1
        if n_frames_load>(n_frames_total-1):
            n_frames_load=(n_frames_total-1)
        t_len = n_frames_load + tG - 1
        g_loss_total,d_loss_total = 0,0
        for idx, data_list in enumerate(tqdm(train_iterator)):
            # if idx==(len(train_iterator)-1):
            #     break
            # with torch.cuda.amp.autocast(enabled=True):
            for i in range(0, n_frames_total-n_frames_load):
                _, _, h, w = data_list['B'].size()
                input_A = (data_list['A'][:, i * input_nc:(i + t_len) * input_nc, ...]).view(-1, t_len, input_nc, h,w)
                input_B = (data_list['B'][:, i * output_nc:(i + t_len) * output_nc, ...]).view(-1, t_len, output_nc,h, w)
                bg_B=(data_list['B'][:, n_frames_total * output_nc, ...]).view(-1, 1, output_nc,h, w)
                bg_A=(data_list['A'][:, n_frames_total * output_nc, ...]).view(-1, 1, output_nc,h, w)
                composite_mask = (torch.max(bg_A) - bg_A)/torch.max(bg_A)
                bg_B[composite_mask == 0] = torch.min(bg_B)
                bg_ref = bg_B.to(device=device, dtype=torch.float32)
                input_A=input_A.to(device=device, dtype=torch.float32)
                input_B = input_B.to(device=device, dtype=torch.float32)

                fake_B_pyr = input_B[:,:(tG-1),...].clone()
                # output_B=input_B[:,:(tG-1),...].clone()
                for t in range(n_frames_load):
                    real_A = 0.4*input_A[:, t:t + tG, ...]/torch.max(input_A[:, t:t + tG, ...])
                    real_B = input_B[:, t:t + tG, ...]
                    real_As_selected = real_A.view(args.train_batch_size, -1, h, w)
                    bg_ref=bg_ref.view(args.train_batch_size, -1, h, w)
                    if t<2:
                        fake_B_prev = fake_B_pyr[:, t: t + tG - 1, ...]
                    else:
                        fake_B_prev = fake_B_pyr[:, 1: 1 + tG - 1, ...]
                    if (t % n_frames_bp) == 0:
                        fake_B_prev = fake_B_prev.detach()
                    fake_B_prev_selected=fake_B_prev.clone().view(args.train_batch_size, -1, h, w)

                    # composite_image = real_A[:,-2,...] + (0.4 - real_A[:,-2,...]) * 0.2*fake_B_prev_selected
                    if t==0:
                        composite_mask=(real_A[:,-2,...] - real_A[:,-2,...])/torch.max(real_A)
                        fake_B_prev_selected[composite_mask == 0] = torch.min(fake_B_prev_selected)
                    # else:
                    #     composite_mask=(torch.max(real_A) - real_A[:,-2,...])/torch.max(real_A)

                    # fake_B_prev_selected[composite_mask == 0] = torch.min(fake_B_prev_selected)
                    input_A[:, t, ...]=fake_B_prev_selected
                    # input_A[:, t, ...]=fake_B_prev_selected[composite_mask==0]=torch.min(fake_B_prev_selected)
                    if random.random() < 0.9:
                        style = mixed_list(fake_B_prev_selected.shape[0], 5, Gen.module.latent_dim, device=device)
                    else:
                        style = noise_list(fake_B_prev_selected.shape[0], 5, Gen.module.latent_dim, device=device)

                    im_noise = image_noise(fake_B_prev_selected.shape[0], image_size, device=device)
                    fake_B = Gen(real_As_selected,fake_B_prev_selected,bg_ref,style, im_noise)
                    fake_B_pyr=torch.cat([fake_B_prev, fake_B.unsqueeze(1)], dim=1)
                    if t == 0:
                        output_B = fake_B.unsqueeze(1).clone()
                    else:
                        output_B = torch.cat([output_B, fake_B.unsqueeze(1)], dim=1)

                    fake_B_current = fake_B_pyr[:, -1, ...]

                    real_B_prev, real_B_current = real_B[:, -2, ...], real_B[:, -1, ...]

                    # real_A_current=real_A[:, -1, ...]

                    real_B_current=real_B_current.contiguous().view(-1, 1, h, w)
                    real_B_prev=real_B_prev.contiguous().view(-1, 1, h, w)
                    fake_B_current=fake_B_current.contiguous().view(-1, 1, h, w)
                    # real_A_current=real_A_current.contiguous().view(-1, 1, h, w)

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

                    Vgg_img_loss=Vgg_criterion(fake_B,real_B_current)
                    L1_img_loss = L1_criterion(fake_B, real_B_current)
                    aug_real_A,aug_fake_B=DiffAugment(real_A.view(-1, 2, h, w),fake_B_pyr.view(-1, 2, h, w),p=p_diff)
                    real_AB = torch.cat((real_A.contiguous().view(-1, 2, h, w), real_B.contiguous().view(-1, 2, h, w)), dim=1)
                    aug_fake_AB = torch.cat((aug_real_A,aug_fake_B), dim=1)
                    valid = torch.full((real_AB.shape[0], 1, 30, 46), 1.0, dtype=real_AB.dtype, device=device)
                    fake = torch.full((real_AB.shape[0], 1, 30, 46), 0.0, dtype=real_AB.dtype, device=device)

                    d_img_loss = d_criterion(D(aug_fake_AB), valid)
                    # if t==0:
                    #     g_loss = d_img_loss + 50*Vgg_img_loss+50*L1_img_loss+150 * flow_loss + 50 * loss_temporal
                    # else:
                    g_loss = d_img_loss + 100 * Vgg_img_loss + 0 * L1_img_loss + 100 * flow_loss + 50 * loss_temporal
                    g_loss_total+=g_loss
                    grad_scaler.scale(g_loss).backward()  # Scale the loss, and then backward pass
                    grad_scaler.step(optimizer_G)  # Update optimizer with scaled gradients
                    grad_scaler.update()  # Update the scale for next iteration
                    scheduler_G.step()

                    set_requires_grad(D, True)
                    optimizer_D.zero_grad(set_to_none=True)
                    aug_real_A,aug_fake_B=DiffAugment(real_A.view(-1, 2, h, w),fake_B_pyr.view(-1, 2, h, w),p=p_diff)
                    aug_fake_AB = torch.cat((aug_real_A,aug_fake_B), dim=1)
                    aug_real_A,aug_real_B=DiffAugment(real_A.view(-1, 2, h, w),real_B.view(-1, 2, h, w),p=p_diff)
                    aug_real_AB = torch.cat((aug_real_A,aug_real_B), dim=1)

                    real_img_loss = d_criterion(D(aug_real_AB), valid)
                    fake_img_loss = d_criterion(D(aug_fake_AB.detach()), fake)
                    d_img_loss = (real_img_loss + fake_img_loss) / 2
                    d_loss_total+=d_img_loss

                    grad_scaler.scale(d_img_loss).backward()
                    grad_scaler.step(optimizer_D)
                    grad_scaler.update()
                    scheduler_D.step()

            # break
            # make_video(input_B,f'frames_real_B/real_B_epoch_idx_{idx}.mp4')
        make_video(input_B, f'frames_real_B_21/real_B_epoch_idx_{idx}.mp4')
        make_video(input_A, f'frames_real_A_21/real_A_epoch_idx_{idx}.mp4')
        make_video(output_B,f'frames_fake_B_21/fake_B_epoch_idx_{idx}.mp4')
        print(Vgg_img_loss.item(), L1_img_loss.item(), flow_loss.item(), loss_temporal.item())
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.max_epoch, d_loss_total.item(), g_loss_total.item())
        )

        # if (epoch % 10 == 0 and epoch < 500) or (epoch > 500 and g_loss < g_loss_min):
        if (g_loss_total < g_loss_min):
            g_loss_min = g_loss_total
            # if epoch > 500:
            #     g_loss_min = g_loss
            torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'Gen.pth'))
            torch.save(D.state_dict(), os.path.join(args.output_dir, 'D.pth'))
            torch.save(FlowNet.state_dict(), os.path.join(args.output_dir, 'FlowNet.pth'))
            Gen_dev.eval()
            Gen_dev.load_state_dict(torch.load(os.path.join(args.output_dir, 'Gen.pth')))
            with torch.no_grad():
                n_frames_load_test = 15
                t_len_test = n_frames_load_test + tG - 1
                for idx, data_list in enumerate(tqdm(test_iterator)):
                    # if idx == (len(test_iterator) - 1):
                    #     break
                    # with torch.cuda.amp.autocast(enabled=True):
                    i = 0
                    _, _, h, w = data_list['B'].size()
                    input_A = (data_list['A'][:, i * input_nc:(i + t_len_test) * input_nc, ...]).view(-1, t_len_test,
                                                                                                      input_nc, h, w)
                    input_B = (data_list['B'][:, i * output_nc:(i + t_len_test) * output_nc, ...]).view(-1, t_len_test,
                                                                                                        output_nc, h,
                                                                                                        w)
                    bg_B = (data_list['B'][:, n_frames_total * output_nc, ...]).view(-1, 1, output_nc, h, w)
                    bg_A = (data_list['A'][:, n_frames_total * output_nc, ...]).view(-1, 1, output_nc, h, w)

                    composite_mask = (torch.max(bg_A) - bg_A) / torch.max(bg_A)
                    bg_B[composite_mask == 0] = torch.min(bg_B)
                    bg_ref = bg_B.to(device=device, dtype=torch.float32)

                    input_A = input_A.to(device=device, dtype=torch.float32)
                    input_B = input_B.to(device=device, dtype=torch.float32)
                    fake_B_pyr = input_B[:, :(tG - 1), ...].clone()
                    # output_B = input_B[:, :(tG - 1), ...].clone()
                    for t in range(n_frames_load_test):
                        real_A = 0.4 * input_A[:, t:t + tG, ...] / torch.max(input_A[:, t:t + tG, ...])
                        real_As_selected = real_A.view(args.test_batch_size, -1, h, w)
                        bg_ref=bg_ref.view(args.test_batch_size, -1, h, w)

                        if t < 2:
                            fake_B_prev = fake_B_pyr[:, t: t + tG - 1, ...]
                        else:
                            fake_B_prev = fake_B_pyr[:, 1: 1 + tG - 1, ...]
                        if (t % n_frames_bp) == 0:
                            fake_B_prev = fake_B_prev.detach()
                        fake_B_prev_selected = fake_B_prev.clone().view(args.test_batch_size, -1, h, w)
                        if t == 0:
                            composite_mask = (real_A[:, -2, ...] - real_A[:, -2, ...]) / torch.max(real_A)
                            fake_B_prev_selected[composite_mask == 0] = torch.min(fake_B_prev_selected)
                        # else:
                        #     composite_mask = (torch.max(real_A) - real_A[:, -2, ...]) / torch.max(real_A)

                        # fake_B_prev_selected[composite_mask == 0] = torch.min(fake_B_prev_selected)


                        if random.random() < 0.9:
                            style = mixed_list(fake_B_prev_selected.shape[0], 5, Gen_dev.module.latent_dim, device=device)
                        else:
                            style = noise_list(fake_B_prev_selected.shape[0], 5, Gen_dev.module.latent_dim, device=device)

                        im_noise = image_noise(fake_B_prev_selected.shape[0], image_size, device=device)
                        fake_B = Gen_dev(real_As_selected, fake_B_prev_selected, bg_ref,style, im_noise)
                        fake_B_pyr = torch.cat([fake_B_prev, fake_B.unsqueeze(1)], dim=1)
                        if t == 0:
                            output_B = fake_B.unsqueeze(1).clone()
                        else:
                            output_B = torch.cat([output_B, fake_B.unsqueeze(1)], dim=1)

                    make_video(input_B, f'frames_real_B_21_test/real_B_epoch_idx_{idx}.mp4')
                    make_video(input_A, f'frames_real_A_21_test/real_A_epoch_idx_{idx}.mp4')
                    make_video(output_B, f'frames_fake_B_21_test/fake_B_epoch_idx_{idx}.mp4')


# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()


    # Parse the command-line arguments
    args = ap.parse_args()
    args.train_set_dir='/media/shared/azargari/few-shot-vid2vid/datasets/cell_2/'
    args.lr=1e-4
    args.max_epoch=2000
    args.train_batch_size=4
    args.test_batch_size=1
    args.p_vanilla=0
    args.output_dir='/home/azargari/cell_vid2vid/train_outputs_21'

    # Check if the test set directory exists
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir

    # # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
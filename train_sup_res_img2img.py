# # Import the necessary libraries and modules
import os
import argparse
from sup_res_models import SupResStyleGenerator,NLayerDiscriminator
from utils import set_requires_grad,image_noise,mixed_list,noise_list
from sup_res_data import BasicDataset
import itertools
import torch.nn as nn
from loss import VGGLoss
import torch.utils.data as data
import sup_res_transforms as transforms
import torch
import numpy as np
import random
import logging
from tqdm import tqdm
from sup_res_diffaug import DiffAugment

# Set a constant seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# The main training function
def train(args,res_in = [256,384],res_out = [512,768],image_means = [0.5],image_stds= [0.5]):
    # Set up the logging
    logging.basicConfig(filename=os.path.join(args.output_dir, 'train.log'), filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)

    logging.info('>>>> input image size=(%d,%d) , learning rate=%f , batch size=%d' % (res_in[0], res_in[1],args.lr,args.batch_size))

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transforms=transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomApply([transforms.RandomOrder([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.33, contrast=0.33, saturation=0.33, hue=0)],p=0.5),
            transforms.RandomApply([transforms.GaussianBlur((5, 5), sigma=(0.1, 1.0))], p=0.5),
            transforms.RandomApply([transforms.RandomHorizontalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.RandomVerticalFlip(0.5)], p=0.5),
            transforms.RandomApply([transforms.AddGaussianNoise(0., 0.01)], p=0.5),
            transforms.RandomApply([transforms.CLAHE()], p=0.5),
            transforms.RandomApply([transforms.RandomAdjustSharpness(sharpness_factor=2)], p=0.5),
            transforms.RandomApply([transforms.RandomCrop()], p=0.5),
        ])], p=args.p_vanilla),
        transforms.Resize(res_in,res_out),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])

    # Define the datasets for training and validation
    train_data = BasicDataset(os.path.join(args.train_set_dir, 'images'),transforms=train_transforms,n_frames_total=args.n_frames_total)

    # Define the dataloaders
    train_iterator = data.DataLoader(train_data,shuffle = True,batch_size = args.batch_size,num_workers=2,pin_memory=True)

    # Define the models
    Gen = SupResStyleGenerator(style_latent_dim = 128)

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

    # Move everything to the device
    Gen = nn.DataParallel(Gen.to(device))
    D = nn.DataParallel(D.to(device))

    d_criterion=d_criterion.to(device)
    Vgg_criterion=Vgg_criterion.to(device)
    L1_criterion = L1_criterion.to(device)

    # Training loop
    logging.info('>>>> Start training')
    print('INFO: Start training ...')
    for epoch in range(args.max_epoch):
        Gen.train()
        D.train()
        g_loss_total,d_loss_total=0,0
        for idx, data_list in enumerate(tqdm(train_iterator)):
            for i in range(0, args.n_frames_total):
                _, _, h, w = data_list['A'].size()
                input_A = (data_list['A'][:, i:(i + 1), ...]).view(-1, 1, h, w)
                _, _, h, w = data_list['B'].size()
                input_B = (data_list['B'][:, i:(i + 1), ...]).view(-1, 1, h, w)

                input_A = input_A.to(device=device, dtype=torch.float32)
                input_B = input_B.to(device=device, dtype=torch.float32)

                if random.random() < 0.9:
                    style = mixed_list(input_B.shape[0], 6, Gen.module.latent_dim, device=device)
                else:
                    style = noise_list(input_B.shape[0], 6, Gen.module.latent_dim, device=device)

                im_noise = image_noise(input_B.shape[0], (h, w), device)

                fake_B = Gen(input_A, style, im_noise)

                real_B = input_B.contiguous().view(-1, 1, h, w)

                fake_B = fake_B.contiguous().view(-1, 1, h, w)

                set_requires_grad(D, False)
                set_requires_grad(Gen, True)

                optimizer_G.zero_grad(set_to_none=True)

                Vgg_img_loss = Vgg_criterion(fake_B, real_B)
                L1_img_loss = L1_criterion(fake_B, real_B)

                valid = torch.full((real_B.shape[0], 1, 62, 94), 1.0, dtype=real_B.dtype, device=device)
                fake = torch.full((real_B.shape[0], 1, 62, 94), 0.0, dtype=real_B.dtype, device=device)

                d_g_img_loss = d_criterion(D(DiffAugment(fake_B,p=args.p_diff)), valid)

                g_loss = d_g_img_loss + 50 * Vgg_img_loss + 25 * L1_img_loss
                g_loss_total+=g_loss

                grad_scaler.scale(g_loss).backward()  # Scale the loss, and then backward pass
                grad_scaler.step(optimizer_G)  # Update optimizer with scaled gradients
                grad_scaler.update()  # Update the scale for next iteration
                scheduler_G.step()

                set_requires_grad(D, True)
                optimizer_D.zero_grad(set_to_none=True)

                real_img_loss = d_criterion(D(DiffAugment(real_B,p=args.p_diff)), valid)
                fake_img_loss = d_criterion(D(DiffAugment(fake_B.detach(),p=args.p_diff)), fake)
                d_img_loss = (real_img_loss + fake_img_loss) / 2
                d_loss_total+=d_img_loss
                grad_scaler.scale(d_img_loss).backward()
                grad_scaler.step(optimizer_D)
                grad_scaler.update()
                scheduler_D.step()


        print(Vgg_img_loss.item(), L1_img_loss.item())
        print(
            "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, args.max_epoch, d_loss_total, g_loss_total)
        )
        logging.info("[Epoch %d/%d] [D loss: %f] [G loss: %f]"% (epoch, args.max_epoch, d_loss_total, g_loss_total))

        if epoch % 10 == 0:
            torch.save(Gen.state_dict(), os.path.join(args.output_dir, 'sup_res_Gen.pth'))
            torch.save(D.state_dict(), os.path.join(args.output_dir, 'sup_res_D.pth'))


# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_set_dir", required=True, type=str, help="path for the train set")
    ap.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    ap.add_argument("--max_epoch", default=2500, type=int, help="maximum epoch to train model")
    ap.add_argument("--batch_size", default=2, type=int, help="train batch size")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the train log and checkpoint")
    ap.add_argument("--p_vanilla", default=0.2, type=float, help="probability value of vanilla augmentation")
    ap.add_argument("--p_diff", default=0.2, type=float,help="probability value of diff augmentation, a value between 0 and 1")
    ap.add_argument("--n_frames_total", default=20, type=int, help="total number of frames to load from each clip")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.train_set_dir), 'No such file or directory: ' + args.train_set_dir
    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    train(args)
# # Import the necessary libraries and modules
import os
import argparse
from low_res_models import Vid2VidStyleGenerator
from sup_res_models import SupResStyleGenerator
from utils import image_noise,mixed_list,noise_list,inception_init,\
    normalize_image,extract_frames,make_test_video,\
    calculate_lpips,calculate_ssim,psnr,calculate_fid
from end2end_data import BasicDataset
import lpips
import torch.nn as nn
import torch.utils.data as data
import low_res_transforms as transforms
import numpy as np
import random
from tqdm import tqdm
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


# The main training function
def test(args,res_in = [256,384],res_out=[512,768],image_means = [0.5],image_stds= [0.5]):

    # Determine the device (GPU or CPU) to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transforms_in = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(res_in),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means,std=image_stds)
    ])
    transforms_out = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(res_out),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means,std=image_stds)
    ])


    test_data = BasicDataset(os.path.join(args.test_set_dir, 'seg_maps'),
                             os.path.join(args.test_set_dir, 'images'), transforms_in=transforms_in,
                             transforms_out=transforms_out,n_frames_total=args.n_frames_total)

    # Define the dataloaders
    test_iterator = data.DataLoader(test_data,batch_size = args.batch_size,num_workers=2 ,pin_memory=True)

    # Define the models
    low_res_Gen = Vid2VidStyleGenerator(style_latent_dim = 128)
    sup_res_Gen=SupResStyleGenerator(style_latent_dim = 128)
    low_res_Gen = nn.DataParallel(low_res_Gen.to(device))
    sup_res_Gen = nn.DataParallel(sup_res_Gen.to(device))
    low_res_Gen.load_state_dict(torch.load(os.path.join(args.low_res_gen_ckpt_dir, 'low_res_Gen.pth')))
    sup_res_Gen.load_state_dict(torch.load(os.path.join(args.sup_res_gen_ckpt_dir, 'sup_res_Gen.pth')))


    low_res_Gen.eval()
    sup_res_Gen.eval()


    inception_model=inception_init(device)

    # Initialize the LPIPS model
    lpips_model = lpips.LPIPS(net='vgg')  # Using AlexNet; you can choose other networks like VGG
    lpips_model = lpips_model.to(device)

    tG=2
    input_nc=1
    output_nc=1
    with torch.no_grad():
        # In your evaluation loop
        fid_score_list,ssim_score_list,psnr_score_list,lpips_score_list,temporal_coherence_score_list=[],[],[],[],[]
        for idx, data_list in enumerate(tqdm(test_iterator)):
            ref_list = data_list.copy()

            n_frames_load_test = data_list['A'].size()[1]-1
            t_len_test = data_list['A'].size()[1]
            real_features_list = []
            fake_features_list = []
            real_frames_list,generated_frames_list=[],[]
            _, _, hi, wi = data_list['A'].size()
            input_A = (data_list['A'][:, 0:t_len_test* input_nc, ...]).view(-1, t_len_test, input_nc, hi, wi)
            input_B = (data_list['B'][:, 0:t_len_test* output_nc, ...]).view(-1, t_len_test, output_nc, hi,wi)
            _, _, ho, wo = data_list['C'].size()
            input_C = (data_list['C'][:, 0:t_len_test* input_nc, ...]).view(-1, t_len_test, input_nc, ho, wo)

            bg_B = (ref_list['B'][:, 3 * output_nc, ...]).view(-1, 1, output_nc, hi, wi)
            bg_A = (ref_list['A'][:, 3 * output_nc, ...]).view(-1, 1, output_nc, hi, wi)

            composite_mask = (torch.max(bg_A) - bg_A) / torch.max(bg_A)
            bg_B[composite_mask == 0] = torch.min(bg_B)
            bg_ref = bg_B.to(device=device, dtype=torch.float32)
            input_A = input_A.to(device=device, dtype=torch.float32)
            input_B = input_B.to(device=device, dtype=torch.float32)
            input_C = input_C.to(device=device, dtype=torch.float32)
            fake_B_pyr = input_B[:, :(tG - 1), ...].clone()

            for t in range(n_frames_load_test):
                real_A = 0.4 * input_A[:, t:t + tG, ...] / torch.max(input_A[:, t:t + tG, ...])
                real_As_selected = real_A.view(args.batch_size, -1, hi, wi)
                bg_ref = bg_ref.view(args.batch_size, -1, hi, wi)
                if t < 2:
                    fake_B_prev = fake_B_pyr[:, t: t + tG - 1, ...]
                else:
                    fake_B_prev = fake_B_pyr[:, 1: 1 + tG - 1, ...]

                fake_B_prev = fake_B_prev.detach()
                fake_B_prev_selected = fake_B_prev.clone().view(args.batch_size, -1, hi, wi)

                if t == 0:
                    composite_mask = (real_A[:, -2, ...] - real_A[:, -2, ...]) / torch.max(real_A)
                    fake_B_prev_selected[composite_mask == 0] = torch.min(input_B)

                if random.random() < 0.9:
                    style = mixed_list(fake_B_prev_selected.shape[0], 5, low_res_Gen.module.latent_dim, device=device)
                else:
                    style = noise_list(fake_B_prev_selected.shape[0], 5, low_res_Gen.module.latent_dim, device=device)

                im_noise = image_noise(fake_B_prev_selected.shape[0], res_in, device=device)
                low_res_fake_B = low_res_Gen(real_As_selected, fake_B_prev_selected, bg_ref, style, im_noise)
                fake_B_pyr = torch.cat([fake_B_prev, low_res_fake_B.unsqueeze(1)], dim=1)
                if t == 0:
                    output_B = low_res_fake_B.unsqueeze(1).clone()
                else:
                    output_B = torch.cat([output_B, low_res_fake_B.unsqueeze(1)], dim=1)
                if random.random() < 0.9:
                    style = mixed_list(fake_B_prev_selected.shape[0], 6, sup_res_Gen.module.latent_dim, device=device)
                else:
                    style = noise_list(fake_B_prev_selected.shape[0], 6, sup_res_Gen.module.latent_dim, device=device)

                im_noise = image_noise(fake_B_prev_selected.shape[0], res_out, device=device)


                low_res_fake_B=normalize_image(low_res_fake_B.detach().cpu().numpy())
                low_res_fake_B=np.transpose(low_res_fake_B.squeeze(0), (1, 2, 0))

                low_res_fake_B,_=transforms_in([low_res_fake_B], [low_res_fake_B])
                low_res_fake_B=low_res_fake_B[0].view(args.batch_size, 1, hi, wi)
                low_res_fake_B=low_res_fake_B.to(device=device, dtype=torch.float32)
                fake_output=sup_res_Gen(low_res_fake_B, style, im_noise)
                if t == 0:
                    output = fake_output.unsqueeze(1).clone()
                else:
                    output = torch.cat([output, fake_output.unsqueeze(1)], dim=1)

            real_frames = extract_frames(input_C[:,1:,...])  # Implement this function
            generated_frames = extract_frames(output)
            real_frames_list.extend(real_frames)
            generated_frames_list.extend(generated_frames)
            if not os.path.exists(f'{args.output_dir}/seq_{idx}/gt_masks/'):
                os.makedirs(f'{args.output_dir}/seq_{idx}/gt_masks/')
            if not os.path.exists(f'{args.output_dir}/seq_{idx}/gt_imgs/'):
                os.makedirs(f'{args.output_dir}/seq_{idx}/gt_imgs/')

            if not os.path.exists(f'{args.output_dir}/seq_{idx}/sup_res_gen_imgs/'):
                os.makedirs(f'{args.output_dir}/seq_{idx}/sup_res_gen_imgs/')

            if not os.path.exists(f'{args.output_dir}/seq_{idx}/low_res_gen_imgs/'):
                os.makedirs(f'{args.output_dir}/seq_{idx}/low_res_gen_imgs/')

            make_test_video(input_A[:,1:,:,:], f'{args.output_dir}/seq_{idx}/gt_masks/',device)
            make_test_video(output_B, f'{args.output_dir}/seq_{idx}/low_res_gen_imgs/', device)
            real_features=make_test_video(input_C[:,1:,:,:], f'{args.output_dir}/seq_{idx}/gt_imgs/',device,model=inception_model)
            fake_features=make_test_video(output, f'{args.output_dir}/seq_{idx}/sup_res_gen_imgs/',device,model=inception_model)
            real_features_list.append(real_features)
            fake_features_list.append(fake_features)
            # Calculate LPIPS for this batch
            batch_lpips_score = calculate_lpips(lpips_model, real_frames, generated_frames, device)
            lpips_score_list.append(batch_lpips_score)

        batch_ssim_score = calculate_ssim(real_frames_list, generated_frames_list)
        ssim_score_list.append(batch_ssim_score)

        # Calculate PSNR for each frame and take the average
        batch_psnr_scores = [psnr(real_frame, gen_frame) for real_frame, gen_frame in
                             zip(real_frames_list, generated_frames_list)]
        avg_batch_psnr = np.mean(batch_psnr_scores)
        psnr_score_list.append(avg_batch_psnr)

        # Combine all batches
        real_features_all = np.concatenate(real_features_list, axis=0)
        fake_features_all = np.concatenate(fake_features_list, axis=0)
        fid_score = calculate_fid(real_features_all, fake_features_all)
        fid_score_list.append(fid_score)

    print(f"FVD Score: {np.mean(fid_score)}")
    print(f"Average SSIM Score: {np.mean(ssim_score_list)}")
    print(f"Average PSNR Score: {np.mean(psnr_score_list)}")
    print(f"Average LPIPS Score: {np.mean(lpips_score_list)}")

# Define the main function
if __name__ == "__main__":
    # Define the argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test set")
    ap.add_argument("--batch_size", default=1, type=int, help="test batch size")
    ap.add_argument("--low_res_gen_ckpt_dir", required=True, type=str, help="path for the low resolution generator checkpoint")
    ap.add_argument("--sup_res_gen_ckpt_dir", required=True, type=str, help="path for the super resolution generator checkpoint")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the outputs")
    ap.add_argument("--n_frames_total", default=20, type=int, help="total number of frames to load from each clip")

    # Parse the command-line arguments
    args = ap.parse_args()

    # Check if the test set directory exists
    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir
    # If output directory doesn't exist, create it
    os.makedirs(args.output_dir, exist_ok=True)

    test(args)
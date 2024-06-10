import random
import cv2
import numpy as np
import torch
from PIL import Image
import os
import torch.nn.functional as F
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
from skimage.metrics import structural_similarity as ssim

DATA_LOADER_SEED = 9
random.seed(DATA_LOADER_SEED)
class_colors = [(0,0,0)]+[(random.randint(50, 255), random.randint(
    50, 255), random.randint(50, 255)) for _ in range(1000)]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c)*(colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c)*(colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c)*(colors[c][2])).astype('uint8')

    return seg_img

def overlay_seg_image(inp_img, seg_img):
    orininal_h = inp_img.shape[0]
    orininal_w = inp_img.shape[1]
    if len(inp_img.shape)==2:
        inp_img=cv2.cvtColor(inp_img, cv2.COLOR_GRAY2BGR)
    seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    fused_img = (inp_img/2 + seg_img/2).astype('uint8')
    return fused_img

def get_legends(class_names, colors=class_colors):

    n_classes = len(class_names)
    legend = np.zeros(((len(class_names) * 25) + 25, 125, 3),
                      dtype="uint8") + 255

    class_names_colors = enumerate(zip(class_names[:n_classes],
                                       colors[:n_classes]))

    for (i, (class_name, color)) in class_names_colors:
        color = [int(c) for c in color]
        cv2.putText(legend, class_name, (5, (i * 25) + 17),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        cv2.rectangle(legend, (100, (i * 25)), (125, (i * 25) + 25),
                      tuple(color), -1)

    return legend

def concat_lenends(seg_img, legend_img):

    new_h = np.maximum(seg_img.shape[0], legend_img.shape[0])
    new_w = seg_img.shape[1] + legend_img.shape[1]

    out_img = np.zeros((new_h, new_w, 3)).astype('uint8') + legend_img[0, 0, 0]

    out_img[:legend_img.shape[0], :  legend_img.shape[1]] = np.copy(legend_img)
    out_img[:seg_img.shape[0], legend_img.shape[1]:] = np.copy(seg_img)

    return out_img

def visualize_segmentation(seg_arr, inp_img=None, n_classes=None,
                           colors=class_colors, class_names=None,
                           overlay_img=False, show_legends=False,
                           prediction_width=None, prediction_height=None):

    if n_classes is None:
        n_classes = np.max(seg_arr)+1

    seg_img = get_colored_segmentation_image(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        orininal_h = inp_img.shape[0]
        orininal_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (orininal_w, orininal_h))

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height))
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    if overlay_img:
        assert inp_img is not None
        seg_img = overlay_seg_image(inp_img, seg_img)

    if show_legends:
        assert class_names is not None
        legend_img = get_legends(class_names, colors=colors)
        seg_img = concat_lenends(seg_img, legend_img)

    return seg_img

def calculate_fid(images1, images2,device):
    # Convert images to PyTorch tensors if they are not
    if not torch.is_tensor(images1):
        images1 = torch.tensor(images1)
    if not torch.is_tensor(images2):
        images2 = torch.tensor(images2)

    # Ensure the images tensors are float and on the right device
    images1 = images1.float().to(device)
    images2 = images2.float().to(device)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)  # Move model to device

    # Extract features for both image sets
    with torch.no_grad():
        pred1 = inception_model(images1)
        pred2 = inception_model(images2)

    # Compute mean and covariance for both sets
    mu1, sigma1 = pred1.mean(0), torch_cov(pred1)
    mu2, sigma2 = pred2.mean(0), torch_cov(pred2)

    # Compute sum of squared difference between the means
    ssdiff = torch.sum((mu1 - mu2)**2.0)

    # Compute sqrt of product between cov
    covmean = sqrtm((sigma1.cpu().numpy() + sigma2.cpu().numpy()) / 2)

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the final FID score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2*torch.tensor(covmean).to(sigma1.device))

    return fid.item()

def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a
            variable, while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def noise(n, latent_dim, device):
    return torch.randn(n, latent_dim).cuda(device)

def noise_list(n, layers, latent_dim, device):
    return [(noise(n, latent_dim, device), layers)]

def mixed_list(n, layers, latent_dim, device):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim, device) + noise_list(n, layers - tt, latent_dim, device)

def image_noise(n, im_size, device):
    return torch.FloatTensor(n, im_size[0], im_size[1], 1).uniform_(0., 1.).cuda(device)


def prepare_weights(map_height, map_width,num_regions):
    # Flat or colony-like with a 50% chance
    if random.random() > 0.5: #Flat
        weights = np.ones((map_height, map_width))
    else: #colony-like
        # Initialize the weights array
        weights = np.zeros((map_height, map_width))
        # Apply random weighted regions
        for _ in range(num_regions):
            # Random center for the circle
            center = (random.randint(0, map_width - 1), random.randint(0, map_height - 1))

            # Random radius
            radius = random.randint(10, 400)  # Adjust min and max radius as needed

            # Random weight increase
            weight_val = 1  # Adjust as needed
            # Apply weights within the circle
            weights = apply_weights_in_circle(weights, center, radius, weight_val,map_height, map_width)

    # Normalize the weights
    flat_weights = weights.flatten()
    flat_weights /= flat_weights.sum()
    return flat_weights

# Function to apply weights within a circle
def apply_weights_in_circle(weights, center, radius, weight_val,map_height, map_width):
    for y in range(max(0, center[1] - radius), min(map_height, center[1] + radius + 1)):
        for x in range(max(0, center[0] - radius), min(map_width, center[0] + radius + 1)):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2:
                weights[y, x] = weight_val
    return weights


def apply_mask_augmentations(mask):
    mask = Image.fromarray(mask)

    # Flip horizontally with a 50% chance
    if random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Flip vertically with a 50% chance
    if random.random() > 0.5:
        mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

    # Rotate randomly between 0 and 360 degrees
    rotation_angle = random.randint(0, 360)
    mask = mask.rotate(rotation_angle, fillcolor=0)  # Assuming mask background is 0 for transparency

    mask = np.array(mask)
    return mask

def save_generated_synthetic_mask(dest_dir, new_mask, mask_id):
    if np.sum(new_mask):
        new_mask[new_mask > 0] = 255
        cv2.imwrite(os.path.join(dest_dir, 'img_{:04d}.png'.format(mask_id)), new_mask)

def apply_cell_body_mask_augmentation(cropped_cell_body):
    # Resize with a 50% chance
    if random.random() > 0.5:#resize
        scale_x = 0.75 + random.randint(0, 50) / 100
        scale_y = 0.75 + random.randint(0, 50) / 100
        cropped_mask = cv2.resize(cropped_cell_body.astype(np.uint8), (0, 0), fx=scale_x, fy=scale_y)
        cropped_mask[cropped_mask < 0.5] = 0
        cropped_mask[cropped_mask >= 0.5] = 1
    return cropped_cell_body


def apply_flow_to_output(image, flow):
    b, _, h, w = image.size()
    x = torch.linspace(0, w - 1, w, device=image.device)
    y = torch.linspace(0, h - 1, h, device=image.device)
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

def normalize_image(image):
    """Normalize image to [0, 1] range."""
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / ((max_val - min_val)+1e-6)

def make_video(input,video_name, fps=1):
    bs, n_f, ch, h, w = input.size()
    input = input[0, :, :, :].view(-1, ch, h, w)
    size = (w, h)
    out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size, isColor=True)

    if not out.isOpened():
        print("Error: Video Writer not initialized.")
        return

    for f_idx in range(n_f):
        frame = input[f_idx, :, :]
        frame = normalize_image(frame.detach().cpu().numpy()) * 255
        frame = frame.astype(np.uint8)
        frame = np.transpose(frame, (1, 2, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()






def calculate_fid(images1, images2,device):
    # Convert images to PyTorch tensors if they are not
    if not torch.is_tensor(images1):
        images1 = torch.tensor(images1)
    if not torch.is_tensor(images2):
        images2 = torch.tensor(images2)

    # Ensure the images tensors are float and on the right device
    images1 = images1.float().to(device)
    images2 = images2.float().to(device)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).eval().to(device)  # Move model to device

    # Extract features for both image sets
    with torch.no_grad():
        pred1 = inception_model(images1)
        pred2 = inception_model(images2)

    # Compute mean and covariance for both sets
    mu1, sigma1 = pred1.mean(0), torch_cov(pred1)
    mu2, sigma2 = pred2.mean(0), torch_cov(pred2)

    # Compute sum of squared difference between the means
    ssdiff = torch.sum((mu1 - mu2)**2.0)

    # Compute sqrt of product between cov
    covmean = sqrtm((sigma1.cpu().numpy() + sigma2.cpu().numpy()) / 2)

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Compute the final FID score
    fid = ssdiff + torch.trace(sigma1 + sigma2 - 2*torch.tensor(covmean).to(sigma1.device))

    return fid.item()

def calculate_fid(real_features, fake_features):
    # Calculate mean and covariance statistics
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    # Calculate FID
    ssdiff = np.sum((mu1 - mu2)**2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

def apply_imagenet_normalization(input):
    r"""Normalize using ImageNet mean and std.

    Args:
        input (4D tensor NxCxHxW): The input images, assuming to be [-1, 1].

    Returns:
        Normalized inputs using the ImageNet normalization.
    """
    # normalize the input back to [0, 1]
    normalized_input = (input + 1) / 2
    # normalize the input using the ImageNet mean and std
    mean = normalized_input.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = normalized_input.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    output = (normalized_input - mean) / std
    return output

def inception_forward(inception, images):
    # images.clamp_(-1, 1)
    images = apply_imagenet_normalization(images)
    # images = F.interpolate(images, size=(299, 299),
    #                        mode='bicubic', align_corners=True)
    return inception(images)


def extract_features(model, input_tensor, device):
    # ImageNet normalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

    # Assuming input_tensor is of shape [batch, sequence, channels, height, width]
    batch, sequence, channels, height, width = input_tensor.shape

    # Process each frame in the sequence individually
    features = []
    for i in range(sequence):
        frame = input_tensor[:, i, ...]  # Select the frame, resulting in a 4D tensor

        # Convert grayscale frame to RGB by replicating the channel
        frame_rgb = frame.repeat(1, 3, 1, 1)  # Repeat the grayscale channel to create RGB channels

        # Resize to 299x299
        # frame_rgb = F.interpolate(frame_rgb, size=(299, 299), mode='bilinear', align_corners=False)

        # Normalize
        frame_rgb = (frame_rgb - mean) / std

        # Extract features
        frame_features = model(frame_rgb.to(device))
        features.append(frame_features)

    # Combine features from all frames
    features = torch.cat(features, dim=0)
    return features.cpu().detach().numpy()

def normalize_image_2(image):
    return image/255*2-1


def make_test_video(input,path,device,model=None):

    bs, n_f, ch, h, w = input.size()
    input = input[0, :, :, :].view(-1, ch, h, w)
    # h, w = 480, 640
    size = (w, h)


    # print("Backend:", out.getBackendName())  # For debugging
    imgs=[]
    for f_idx in range(n_f):
        frame = input[f_idx, :, :]
        frame = (frame.detach().cpu().numpy()+1)/2*255
        frame = np.transpose(frame, (1, 2, 0))
        cv2.imwrite(os.path.join(path, 'img_{:02d}.png'.format(f_idx)), (normalize_image(frame)*255).astype(np.uint8))
        frame=normalize_image_2(frame)
        imgs.append(frame)
    if model is not None:
        imgs=torch.tensor(imgs).to(device=device, dtype=torch.float32).view(1, -1,1, frame.shape[0], frame.shape[1])
        features = extract_features(model, imgs,device=device)
        return features


def calculate_ssim(real_frames, generated_frames):
    ssim_scores = []
    for real_frame, generated_frame in zip(real_frames, generated_frames):
        # SSIM calculation for each pair of frames
        ssim_score = ssim(real_frame[:, :, 0], generated_frame[:, :, 0], data_range=real_frame.max() - real_frame.min(), multichannel=True)
        ssim_scores.append(ssim_score)
    return np.mean(ssim_scores)

def extract_frames(video_tensor):
    """
    Extracts frames from a video tensor and prepares them for SSIM computation.

    Args:
        video_tensor (torch.Tensor): A tensor representing a video.
                                     Expected shape: [batch_size, number_of_frames, channels, height, width]

    Returns:
        List of NumPy arrays representing individual frames.
    """
    frames = []
    for i in range(video_tensor.shape[1]):  # Iterate over each frame
        frame = video_tensor[0, i]  # Assuming batch size is always 1

        # Normalize the frame to [0, 1] range
        min_val = frame.min()
        max_val = frame.max()
        frame_normalized = (frame - min_val) / (max_val - min_val)

        # Convert to NumPy array and add channel dimension if needed
        frame_np = frame_normalized.squeeze().cpu().numpy()
        if frame_np.ndim == 2:  # Add color channel for grayscale images
            frame_np = np.stack((frame_np,)*3, axis=-1)

        frames.append(frame_np)
    return frames


def psnr(img1, img2):
    """
    Calculates the PSNR between two images.

    Args:
        img1 (numpy.ndarray): The first image.
        img2 (numpy.ndarray): The second image.

    Returns:
        float: The PSNR value.
    """
    img1=normalize_image(img1)
    img2 = normalize_image(img2)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming your images are scaled between 0 and 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_lpips(lpips_model, real_frames, generated_frames, device):
    """
    Calculates LPIPS between two sets of frames.

    Args:
        lpips_model: Pretrained LPIPS model.
        real_frames (List[np.ndarray]): List of real frames as numpy arrays.
        generated_frames (List[np.ndarray]): List of generated frames as numpy arrays.
        device: Torch device to use (CPU or CUDA).

    Returns:
        float: The average LPIPS score.
    """
    lpips_scores = []
    for real_frame, gen_frame in zip(real_frames, generated_frames):
        # Convert numpy arrays to torch tensors
        real_tensor = torch.from_numpy(real_frame).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
        gen_tensor = torch.from_numpy(gen_frame).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)

        # Calculate LPIPS
        lpips_score = lpips_model.forward(real_tensor, gen_tensor)
        lpips_scores.append(lpips_score.item())

    return np.mean(lpips_scores)



def inception_init(device):
    inception = inception_v3(pretrained=True, transform_input=False)
    inception = inception.to(device)
    inception.eval()
    inception.fc = torch.nn.Sequential()
    return inception
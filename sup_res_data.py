import cv2
import os.path
import os.path
import torch
import os
import os.path
import numpy as np

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff',
    '.txt', '.json'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        paths = []
        root = fname[0]
        for f in sorted(fname[2]):
            if is_image_file(f):
                paths.append(os.path.join(root, f))
        if len(paths) > 0:
            images.append(paths)
    return images


def concat_frame(A, Ai, nF):
    if A is None:
        A = Ai
    else:
        c = Ai.size()[0]
        if A.size()[0] == nF * c:
            A = A[c:]
        A = torch.cat([A, Ai])
    return A

class BasicDataset():
    def __init__(self, clips_dir,transforms=None,n_frames_total=20):
        self.clips_dir = clips_dir
        self.A_is_label = True
        self.use_instance=True
        self.n_frames_G=2
        self.transforms=transforms
        self.n_frames_total=n_frames_total
        self.clips_paths = sorted(make_grouped_dataset(self.clips_dir))
    def __getitem__(self, seq_idx):

        images_paths = self.clips_paths[seq_idx]
        img_list=[]
        for i in range(self.n_frames_total):
            img_path = images_paths[i]
            img = self.get_image(img_path)
            img_list.append(img)
        A_list, B_list = self.preprocess(img_list,img_list, self.transforms)
        A,B=None,None
        for Ai,Bi in zip(A_list, B_list):
            A = concat_frame(A, Ai, self.n_frames_total)
            B = concat_frame(B, Bi, self.n_frames_total)

        return_list = {'A': A, 'B': B}
        return return_list

    def preprocess(self, pil_img, pil_mask,transforms):
        tensor_img,tensor_mask=transforms(pil_img,pil_mask)
        return tensor_img,tensor_mask

    def get_image(self, img_path):
        img = cv2.imread(img_path, 0).astype('float32')
        img = (255 * ((img - img.min()) / (img.ptp() + 1e-6))).astype(np.uint8)
        return img

    def __len__(self):
        return len(self.clips_paths)







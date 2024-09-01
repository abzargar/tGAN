import cv2
from torch.utils.data import Dataset
from pathlib import Path
import os.path
import torchvision.transforms as transforms
import random
import os.path
import torch
import torch.utils.data as data
import os
import os.path
from PIL import Image
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

def check_path_valid(A_paths, B_paths):
    assert(len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert(len(a) == len(b))

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
    def __init__(self, seg_map_seqs_dir,images_seqs_dir,transforms=None,n_frames_total=20):
        self.seg_map_seqs_dir = seg_map_seqs_dir
        self.images_seqs_dir = images_seqs_dir
        self.n_frames_G=2
        self.transforms=transforms
        self.n_frames_total=n_frames_total
        self.seg_map_seqs_dir = sorted(make_grouped_dataset(self.seg_map_seqs_dir))
        self.images_seqs_dir = sorted(make_grouped_dataset(self.images_seqs_dir))
        check_path_valid(self.seg_map_seqs_dir, self.images_seqs_dir)
        self.seq_idx = 0  # index for current sequence
        self.frame_idx = 0  # index for current frame in the sequence
        self.A, self.B= None, None

    def __getitem__(self, seq_idx):

        seg_maps_paths = self.seg_map_seqs_dir[seq_idx]
        images_paths = self.images_seqs_dir[seq_idx]
        n_frames_total = min(self.n_frames_total, len(seg_maps_paths) - self.n_frames_G + 1)
        n_frames_total = n_frames_total + self.n_frames_G - 1  # rounded overall number of frames to read from the sequence
        frame_range = list(range(n_frames_total)) if self.A is None else [self.n_frames_G - 1]
        A_list,B_list=[],[]
        for i in frame_range:
            seg_map_path = seg_maps_paths[i]
            image_path = images_paths[i]
            Ai,Bi = self.get_image(seg_map_path, image_path)
            A_list.append(Ai)
            B_list.append(Bi)

        B_list, A_list = self.preprocess(B_list, A_list, self.transforms)
        A,B=None,None
        for Bi,Ai in zip(B_list, A_list):
            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)

        return_list = {'A': A, 'B': B,'A_path': seg_map_path}
        return return_list

    def preprocess(self, pil_img, pil_mask,transforms):
        tensor_img,tensor_mask=transforms(pil_img,pil_mask)
        return tensor_img,tensor_mask

    def get_image(self, A_path, B_path):
        A_img = cv2.imread(A_path, 0) > 0
        A_img = A_img.astype('float32')
        B_img = cv2.imread(B_path, 0).astype('float32')
        B_img = (255 * ((B_img - B_img.min()) / (B_img.ptp() + 1e-6))).astype(np.uint8)

        return A_img,B_img

    def __len__(self):
        return len(self.seg_map_seqs_dir)








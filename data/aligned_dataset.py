import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image



class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(self.dir_AB))
        assert(opt.resize_or_crop == 'resize_and_crop')

    # my get_item function
    
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        # Load the 4-channel .tiff image as a tensor directly
        AB = Image.open(AB_path)
        AB_tensor = transforms.ToTensor()(AB)  # Will preserve all channels (e.g., 4)

        if AB_tensor.shape[0] != 4:
            raise ValueError(f"Expected 4-channel image, got {AB_tensor.shape[0]} channels for {AB_path}")

        # Normalize each channel (mean=0.5, std=0.5) assuming 4 channels
        AB_tensor = transforms.Normalize([0.5] * 4, [0.5] * 4)(AB_tensor)

        # Split channels
        T1 = AB_tensor[0, ...]
        T2 = AB_tensor[1, ...]
        FLAIR = AB_tensor[2, ...]
        T1ce = AB_tensor[3, ...]

        # Create inputs and target
        if self.opt.which_direction == 'AtoB':
            A = torch.cat([T1.unsqueeze(0), T2.unsqueeze(0), FLAIR.unsqueeze(0)], dim=0)  # [T1, T2, FLAIR]
            B = T1ce.unsqueeze(0)  # [T1ce]
        elif self.opt.which_direction == 'BtoA':
            A = T1ce.unsqueeze(0)  # [T1ce]
            B = torch.cat([T1.unsqueeze(0), T2.unsqueeze(0), FLAIR.unsqueeze(0)], dim=0)  # [T1, T2, FLAIR]

        # Resize if needed (make sure it's consistent with AB.resize previously)
        A = transforms.functional.resize(A, [self.opt.loadSize, self.opt.loadSize], antialias=True)
        B = transforms.functional.resize(B, [self.opt.loadSize, self.opt.loadSize], antialias=True)


        # Apply random horizontal and vertical flip
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = torch.flip(A, dims=[1])  # Horizontal flip
            B = torch.flip(B, dims=[1])
        if (not self.opt.no_flip) and random.random() < 0.5:
            A = torch.flip(A, dims=[2])  # Vertical flip
            B = torch.flip(B, dims=[2])

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    # def __getitem__(self, index):
    #     AB_path = self.AB_paths[index]
    #     AB = Image.open(AB_path).convert('RGB')  # Open as RGB image (3 channels)
    #     # Alternatively, try flipping if needed
    #     # AB = AB.transpose(Image.FLIP_TOP_BOTTOM)
        
    #     # Resize the image to match the desired input size
    #     AB = AB.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        
    #     # Convert the image to a tensor and normalize it
    #     AB_tensor = transforms.ToTensor()(AB)
    #     AB_tensor = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(AB_tensor)

    #     # Split the channels to form T1, T2, and T1ce
    #     T1 = AB_tensor[0, ...]  # First channel (T1)
    #     T2 = AB_tensor[1, ...]  # Second channel (T2)
    #     T1ce = AB_tensor[2, ...]  # Third channel (T1ce)

    #     # Define input_nc and output_nc based on the direction
    #     if self.opt.which_direction == 'AtoB':
    #         # A should contain T1 and T2 (input), B should contain T1ce (output)
    #         A = torch.cat([T1.unsqueeze(0), T2.unsqueeze(0)], dim=0)  # A = [T1, T2]
    #         B = T1ce.unsqueeze(0)  # B = [T1ce]
    #     elif self.opt.which_direction == 'BtoA':
    #         # A should contain T1ce (input), B should contain T1 and T2 (output)
    #         A = T1ce.unsqueeze(0)  # A = [T1ce]
    #         B = torch.cat([T1.unsqueeze(0), T2.unsqueeze(0)], dim=0)  # B = [T1, T2]

    #     # Apply random flip augmentation if required
    #     if (not self.opt.no_flip) and random.random() < 0.5:
    #         # Flip along the height (axis 1) and width (axis 2)
    #         A = A.flip(1).flip(2)
    #         B = B.flip(1).flip(2)

    #     # Ensure that the image paths are passed as well
    #     return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
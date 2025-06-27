from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_valid_image(file_name):
    valid_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    return any(file_name.endswith(ext) for ext in valid_extensions)


def adjust_crop_size(size, scale):
    return size - (size % scale)


def high_res_transform(crop_dim):
    return Compose([
        RandomCrop(crop_dim),
        ToTensor(),
    ])


def low_res_transform(crop_dim, scale):
    return Compose([
        ToPILImage(),
        Resize(crop_dim // scale, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def visual_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainingSetLoader(Dataset):
    def __init__(self, data_root, crop_dim, scale_factor):
        super(TrainingSetLoader, self).__init__()
        self.img_paths = [join(data_root, f) for f in listdir(data_root) if is_valid_image(f)]
        crop_dim = adjust_crop_size(crop_dim, scale_factor)
        self.hr_proc = high_res_transform(crop_dim)
        self.lr_proc = low_res_transform(crop_dim, scale_factor)

    def __getitem__(self, idx):
        high_res_img = self.hr_proc(Image.open(self.img_paths[idx]))
        low_res_img = self.lr_proc(high_res_img)
        return low_res_img, high_res_img

    def __len__(self):
        return len(self.img_paths)


class ValidationSetLoader(Dataset):
    def __init__(self, val_data_dir, scale_factor):
        super(ValidationSetLoader, self).__init__()
        self.scale_factor = scale_factor
        self.val_img_paths = [join(val_data_dir, f) for f in listdir(val_data_dir) if is_valid_image(f)]

    def __getitem__(self, idx):
        high_res_img = Image.open(self.val_img_paths[idx])
        width, height = high_res_img.size
        crop_size = adjust_crop_size(min(width, height), self.scale_factor)
        lr_resize = Resize(crop_size // self.scale_factor, interpolation=Image.BICUBIC)
        hr_resize = Resize(crop_size, interpolation=Image.BICUBIC)

        high_res_img = CenterCrop(crop_size)(high_res_img)
        low_res_img = lr_resize(high_res_img)
        restored_hr_img = hr_resize(low_res_img)

        return ToTensor()(low_res_img), ToTensor()(restored_hr_img), ToTensor()(high_res_img)

    def __len__(self):
        return len(self.val_img_paths)


class TestingSetLoader(Dataset):
    def __init__(self, test_root_dir, scale_factor):
        super(TestingSetLoader, self).__init__()
        self.scale_factor = scale_factor
        self.lr_dir = join(test_root_dir, f'SRF_{scale_factor}', 'data')
        self.hr_dir = join(test_root_dir, f'SRF_{scale_factor}', 'target')

        self.lr_files = [join(self.lr_dir, f) for f in listdir(self.lr_dir) if is_valid_image(f)]
        self.hr_files = [join(self.hr_dir, f) for f in listdir(self.hr_dir) if is_valid_image(f)]

    def __getitem__(self, idx):
        file_name = self.lr_files[idx].split('/')[-1]

        low_res_img = Image.open(self.lr_files[idx])
        hr_w, hr_h = low_res_img.size

        high_res_img = Image.open(self.hr_files[idx])
        resize_hr = Resize((self.scale_factor * hr_h, self.scale_factor * hr_w), interpolation=Image.BICUBIC)
        restored_img = resize_hr(low_res_img)

        return file_name, ToTensor()(low_res_img), ToTensor()(restored_img), ToTensor()(high_res_img)

    def __len__(self):
        return len(self.lr_files)



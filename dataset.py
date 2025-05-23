import json
from pathlib import Path
from typing import List, Optional, Union, Dict, Literal

import PIL
import PIL.Image
from torch.utils.data import Dataset


class COCOCaptionDataset(Dataset):
    """
   CIRR dataset class for PyTorch dataloader.
   The dataset can be used in 'relative' or 'classic' mode:
        - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
        - In 'relative' mode the dataset yield dict with keys:
            - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption', 'group_members']
             when split in ['train', 'val']
            - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
    """

    def __init__(self, dataset_path: Union[Path, str], file_path: Union[Path, str], preprocess: callable):
        """
        :param dataset_path: path to the CIRR dataset
        :param split: dataset split, should be in ['train', 'val', 'test']
        :param mode: dataset mode, should be in ['relative', 'classic']:
                - In 'classic' mode the dataset yield a dict with keys ['image', 'image_name']
                - In 'relative' mode the dataset yield dict with keys:
                    - ['reference_image', 'reference_name', 'target_image', 'target_name', 'relative_caption',
                    'group_members'] when split in ['train', 'val']
                    - ['reference_image', 'reference_name' 'relative_caption', 'group_members', 'pair_id'] when split == test
        :param preprocess: function which preprocesses the image
        :param no_duplicates: if True, the dataset will not yield duplicate images in relative mode, does not affect classic mode
        """
        dataset_path = Path(dataset_path)
        self.dataset_path = dataset_path
        self.file_path = Path(file_path)
        self.preprocess = preprocess

        
        # get triplets made by (reference_image, target_image, relative caption)
        with open(file_path) as f:
            self.annotation = json.load(f)


    def __getitem__(self, index) -> dict:

        image_id = self.annotation[index]['image_id']
        image_name = self.annotation[index]['file_name']
        image_path = self.dataset_path / image_name
        image = self.preprocess(PIL.Image.open(image_path))
        caption = self.annotation[index]["caption"]
        return {
            'image': image,
            'positive_caption': caption,
            'image_id': image_name
        }
           
    def __len__(self):
       
        return len(self.annotation)
    
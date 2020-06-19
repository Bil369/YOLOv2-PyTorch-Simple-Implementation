import os
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from src.data_augmentation import *


class VOCDataset(Dataset):
    def __init__(self, dataset_path, image_size=448, is_training = True):
        self.dataset_path = dataset_path
        self.images = sorted(os.listdir(dataset_path + '/JPEGImages'))
        self.annotations = sorted(os.listdir(dataset_path + '/Annotations'))
        self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                        'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                        'tvmonitor']
        self.image_size = image_size
        self.num_classes = len(self.classes)
        self.num_images = len(self.images)
        self.is_training = is_training

    def __len__(self):
        return self.num_images

    def __getitem__(self, item):
        image_path = os.path.join(self.dataset_path, "JPEGImages", self.images[item])
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_xml_path = os.path.join(self.dataset_path, "Annotations", self.annotations[item])
        annot = ET.parse(image_xml_path)

        objects = []
        for obj in annot.findall('object'):
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in
                                      ["xmin", "xmax", "ymin", "ymax"]]
            label = self.classes.index(obj.find('name').text.lower().strip())
            objects.append([xmin, ymin, xmax, ymax, label])
        if self.is_training:
            transformations = Compose([HSVAdjust(), VerticalFlip(), Crop(), Resize(self.image_size)])
        else:
            transformations = Compose([Resize(self.image_size)])
        image, objects = transformations((image, objects))

        return np.transpose(np.array(image, dtype=np.float32), (2, 0, 1)), np.array(objects, dtype=np.float32)

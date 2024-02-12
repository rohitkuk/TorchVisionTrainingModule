import os
from torchvision import datasets
import torchvision.transforms as transforms


image_transforms = {
    'train': transforms.Compose([
         transforms.Resize((224, 224)),
        # Randomly crop and resize the image
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.5, 0.5, 0.5],  # Normalize the image
                             [0.5, 0.5, 0.5])
    ]),
    'valid': transforms.Compose([
         transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.5, 0.5, 0.5],  # Normalize the image
                             [0.5, 0.5, 0.5])
    ])
}



class ProductsDataset:
    """
    A class for handling the Water Logging dataset.

    Parameters:
        root_dir (str): The root directory of the dataset.
        transforms (torchvision.transforms.Compose): The image transformations to apply.

    Methods:
        __call__(self, dataset_type):
            Loads the specified dataset type.

        validate_dataset_type(self, dataset_type):
            Validates the dataset type and returns the formatted dataset type.

    """

    def __init__(self, root_dir):
        """
        Initializes the WaterLoggingDataset object.

        Args:
            root_dir (str): The root directory of the dataset.

        """
        self.root_dir = root_dir
 

    def __call__(self, dataset_type, transforms):
        """
        Loads the specified dataset type.

        Args:
            dataset_type (str): The type of the dataset to load.

        Returns:
            torchvision.datasets.ImageFolder: The loaded dataset.

        """
        dataset_type = self.validate_dataset_type(dataset_type)
        dataset_dir = os.path.join(self.root_dir, dataset_type)
        dataset = datasets.ImageFolder(
            root=dataset_dir, transform=transforms)
        return dataset

    def validate_dataset_type(self, dataset_type):
        """
        Validates the dataset type and returns the formatted dataset type.

        Args:
            dataset_type (str): The dataset type to validate.

        Returns:
            str: The formatted dataset type.

        Raises:
            Exception: If the dataset type is not valid.

        """
        dataset_type = dataset_type.lower().strip()
        if dataset_type == 'val':
            dataset_type = 'valid'
        valid_types = ['train', 'valid', 'test', 'val']
        if dataset_type not in valid_types:
            raise Exception(f"""
                            Dataset Type {dataset_type} is not a valid dataset type. 
                            Please choose from the following: {valid_types}
                        """)
        return dataset_type

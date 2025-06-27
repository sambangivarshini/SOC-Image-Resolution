# SOC-Image-Resolution

## __data-utils.py__

Preparing and Loading the Dataset
In data_utils.py, I learned how to handle and preprocess datasets for training, validation, and testing in PyTorch:

Using PyTorch Datasets and Dataloaders:
I created custom dataset classes (TrainingSetLoader, ValidationSetLoader, and TestingSetLoader) to efficiently load and batch data.

Data Augmentation:
I applied transformations like RandomCrop, Resize, CenterCrop, and ToTensor to prepare both low-resolution and high-resolution image pairs.

Handling Image Formats:
I learned how to filter valid image files using extensions and how to load them using PIL.

Multi-Scale Support:
I made the data pipeline flexible for different scale factors (e.g., ×2, ×4) so the model can be trained and tested on various upscaling tasks.

Reproducibility:
By using consistent random crops and preprocessing steps, I ensured that the training is reproducible and aligned with academic standards.

## __Model.py__


Designing the Super-Resolution Model:
In model.py, I learned how to architect and implement a Convolutional Neural Network (CNN) using PyTorch for image super-resolution. Key takeaways include:

Model Architecture:
I built a deep network that takes low-resolution images as input and outputs super-resolved images. This helped me understand how to:

Stack multiple convolutional layers.

Apply activation functions like ReLU.

Use upsampling techniques such as sub-pixel convolution or interpolation.

Weight Initialization:
I learned the importance of proper weight initialization (e.g., using He/Kaiming initialization) to stabilize and speed up training.

Forward Pass:
I implemented the forward() function to define how data flows through the layers of the network.

Modularity:
I structured the model in a modular way so it can be easily trained, evaluated, or extended later.

## __loss.py__

Implementing Loss Functions:
In loss.py, I learned how to define and use custom loss functions to train super-resolution models more effectively:

Pixel-wise Loss (L1/L2 Loss):
I implemented losses like Mean Squared Error (MSE) or L1 Loss to measure how close the predicted image is to the ground truth at the pixel level.

Perceptual Loss:
I integrated a perceptual loss using features from a pretrained network (like VGG). This taught me:

How to extract feature maps from intermediate layers of a CNN.

How perceptual loss helps produce visually better images, even if pixel-wise loss is higher.

Combining Losses:
I experimented with weighted combinations of losses (e.g., Total Loss = α × L1 Loss + β × Perceptual Loss) to balance sharpness and accuracy.

PyTorch Autograd Compatibility:
I made sure all custom loss functions work seamlessly with PyTorch’s autograd engine so gradients flow correctly during backpropagation.





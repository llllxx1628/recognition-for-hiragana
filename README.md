# Hiragana Recognition

This project implements a neural network for recognizing Hiragana characters from the Kuzushiji-49 dataset. The model is built using PyTorch and incorporates techniques like depthwise separable convolutions, residual connections, and attention mechanisms (CBAM).

## Features
- Multi-scale convolutional blocks
- Residual connections for better gradient flow
- Channel and spatial attention (CBAM) for enhanced feature representation
- Label smoothing to prevent overconfidence in predictions

## Requirements
The project is build on Python 3.8+ and depends on the following libraries:

```plaintext
torch==2.0.0
torchvision==0.15.1
numpy==1.21.2
Pillow==9.1.0
scikit-learn==1.2.2
```

Install the dependencies using:
```bash
pip install -r requirements.txt
```

## Getting Started
### Step 1: Clone the Repository
```bash
git clone https://github.com/llllxx1628/recognition-for-hiragana.git
cd hiragana-recognition
```

### Step 2: Prepare the Dataset
Ensure that you have downloaded Kuzushiji-49 dataset in the `.npz` format, which is avaliable at https://github.com/rois-codh/kmnist.git:
- `k49-train-imgs.npz` (training images)
- `k49-train-labels.npz` (training labels)
- `k49-test-imgs.npz` (test images)
- `k49-test-labels.npz` (test labels)

Place the dataset files in the appropriate directory.

### Step 3: Run the Training Script
To train the model, use the following command:
```bash
python train_model.py --mode train \
                      --train_images path/to/train_images.npz \
                      --train_labels path/to/train_labels.npz \
                      --test_images path/to/test_images.npz \
                      --test_labels path/to/test_labels.npz \
                      --batch_size 64 \
                      --epochs 50 \
                      --lr 0.001 \
                      --weight_decay 1e-4 \
                      --model_save_path best_model.pth 
```

### Step 4: Evaluate the Model
To evaluate the model on the test dataset, use the following command:
```bash
python train_model.py --mode eval \
                      --test_images path/to/test_images.npz \
                      --test_labels path/to/test_labels.npz \
                      --batch_size 64 \
                      --model_save_path best_model.pth
```

## Code Structure
- `model.py`: Contains the implementation of the neural network architecture and loss function.
- `train_model.py`: Contains the training and validation logic.
- `requirements.txt`: Lists the required dependencies.

## Improvements and Future Initiatives
Given more time, the following initiatives could have been undertaken:
1. **Hyperparameter Tuning**:
   - Optimize learning rates, weight decay, and other hyperparameters for better performance.

2. **Explainability**:
   - Visualize feature maps and attention weights to understand what the model is focusing on.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements
- Kuzushiji-49 dataset: [Kuzushiji Dataset](https://github.com/rois-codh/kmnist)
- PyTorch for providing the framework for building and training the model.


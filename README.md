# Hiragana Recognition

This project implements a neural network for recognizing Hiragana characters from the Kuzushiji-49 dataset. The model is built using PyTorch and incorporates techniques like depthwise separable convolutions, residual connections, and attention mechanisms (CBAM).

## Features
- Multi-scale convolutional blocks
- Residual connections for better gradient flow
- Channel and spatial attention (CBAM) for enhanced feature representation
- Label smoothing to prevent overconfidence in predictions
- Split the training dataset into train/validation sets

## Requirements
The project depends on the following libraries:

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
git clone https://github.com/yourusername/hiragana-recognition.git
cd hiragana-recognition
```

### Step 2: Prepare the Dataset
Ensure that you have the Kuzushiji-49 dataset in the `.npz` format:
- `k49-train-imgs.npz` (training images)
- `k49-train-labels.npz` (training labels)
- `k49-test-imgs.npz` (test images)
- `k49-test-labels.npz` (test labels)

Place the dataset files in the appropriate directory.

### Step 3: Run the Training Script
Train the model with the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python train_model.py \
    --train_images /path/to/k49-train-imgs.npz \
    --train_labels /path/to/k49-train-labels.npz \
    --val_images /path/to/k49-test-imgs.npz \
    --val_labels /path/to/k49-test-labels.npz
```

### Step 4: Evaluate the Model
The script will automatically evaluate the model on the validation set and save the best model as `best_model.pth`.

### Step 5: Testing the Model
You can use the saved model to perform predictions on unseen data. Modify the script to load the model weights and test on your dataset.

## Code Structure
- `model.py`: Contains the implementation of the neural network architecture and loss function.
- `train_model.py`: Contains the training and validation logic.
- `requirements.txt`: Lists the required dependencies.

## Improvements and Future Initiatives
Given more time, the following initiatives could have been undertaken:
1. **Hyperparameter Tuning**:
   - Optimize learning rates, weight decay, and other hyperparameters for better performance.

2. **Data Augmentation**:
   - Add additional augmentation techniques like elastic transformations to further improve generalization.

3. **Fine-Grained Validation**:
   - Use k-fold cross-validation to better evaluate the model's performance.

4. **Explainability**:
   - Visualize feature maps and attention weights to understand what the model is focusing on.

5. **Deployment**:
   - Package the model into an inference service using frameworks like Flask or FastAPI.

6. **Performance Optimization**:
   - Experiment with quantization or pruning to make the model lightweight and deployable on edge devices.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements
- Kuzushiji-49 dataset: [Kuzushiji Dataset](https://github.com/rois-codh/kmnist)
- PyTorch for providing the framework for building and training the model.


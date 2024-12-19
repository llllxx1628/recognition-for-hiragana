import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, RandomRotation, RandomAffine, ColorJitter
from sklearn.metrics import balanced_accuracy_score
from model import HiraganaRecognitionNet, LabelSmoothingLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class KuzushijiDataset(Dataset):
    """
    Dataset class for Kuzushiji-49.
    """

    def __init__(self, image_file, label_file, transform=None):
        self.images = np.load(image_file)['arr_0']
        self.labels = np.load(label_file)['arr_0']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            image = self.transform(image)
        return image, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or Evaluate Hiragana Recognition Model.")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help="Mode: train or eval")
    parser.add_argument('--train_images', type=str, help="Path to training images file")
    parser.add_argument('--train_labels', type=str, help="Path to training labels file")
    parser.add_argument('--test_images', type=str, required=True, help="Path to test images file")
    parser.add_argument('--test_labels', type=str, required=True, help="Path to test labels file")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--early_stop_patience', type=int, default=10, help="Early stopping patience")
    parser.add_argument('--model_save_path', type=str, default="best_model.pth", help="Path to save the best model")
    args = parser.parse_args()

    # train model
    if args.mode == "train":
        dataset = KuzushijiDataset(
            image_file=args.train_images,
            label_file=args.train_labels,
            transform=Compose([
                ToTensor(),
                RandomRotation(degrees=15),
                RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                ColorJitter(0.2, 0.2),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
        )

        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # model initialization
        model = HiraganaRecognitionNet(num_classes=49).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        criterion = LabelSmoothingLoss(smoothing=0.1)

        best_val_loss = float('inf')
        early_stop_counter = 0

        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_loader.dataset)

            model.eval()
            val_loss = 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    preds = outputs.argmax(dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            val_accuracy = balanced_accuracy_score(all_labels, all_preds) * 100

            print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(model.state_dict(), args.model_save_path)
                print("Best model saved.")
            else:
                early_stop_counter += 1
                if early_stop_counter >= args.early_stop_patience:
                    print("Early stopping triggered.")
                    break

    # eval model
    elif args.mode == "eval":
        test_dataset = KuzushijiDataset(
            image_file=args.test_images,
            label_file=args.test_labels,
            transform=Compose([
                ToTensor(),
                Normalize(mean=(0.5,), std=(0.5,))
            ])
        )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        model = HiraganaRecognitionNet(num_classes=49).to(device)
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        model.eval()

        test_loss = 0
        all_preds, all_labels = [], []
        criterion = LabelSmoothingLoss(smoothing=0.1)
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss /= len(test_loader.dataset)
        test_accuracy = balanced_accuracy_score(all_labels, all_preds) * 100
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

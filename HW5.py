import os
import cv2
import torch
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import train_test_split

# Paths to images and masks
# image_path = 'lab5/images'
# mask_path = 'lab5/masks'

image_path = 'lab3/b/Pratheepan_Dataset/FacePhoto'
mask_path = 'lab3/b/Ground_Truth/GroundT_FacePhoto'

image_files = sorted(os.listdir(image_path))  # Sorting ensures matching order
mask_files = sorted(os.listdir(mask_path))


class DatasetClass(Dataset):
    def __init__(self, image_files, mask_files, image_dir, mask_dir, transform):
        super().__init__()
        self.image_files = image_files
        self.mask_files = mask_files
        self.image_dir = image_dir  # Path to the image directory
        self.mask_dir = mask_dir    # Path to the mask directory
        self.transform = transform
        self.length = len(mask_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Construct full paths for the image and mask
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # Open image and mask
        img = Image.open(img_path)
        mask = Image.open(mask_path)

        # Apply transformations
        img = self.transform(img)
        mask = self.transform(mask)

        return img, mask


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Max-pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)  # Input: 1x256x256 --> Output: 64x256x256
        x1_pooled = self.pool(x1)  # 64x256x256 --> 64x128x128

        x2 = self.enc_conv2(x1_pooled)  # 64x128x128 --> 128x128x128
        x2_pooled = self.pool(x2)  # 128x128x128 --> 128x64x64

        x3 = self.enc_conv3(x2_pooled)  # 128x64x64 --> 256x64x64
        x3_pooled = self.pool(x3)  # 256x64x64 --> 256x32x32

        x4 = self.enc_conv4(x3_pooled)  # 256x32x32 --> 512x32x32
        x4_pooled = self.pool(x4)  # 512x32x32 --> 512x16x16

        # Bottleneck
        bottleneck = self.bottleneck(x4_pooled)  # 512x16x16 --> 1024x16x16

        # Decoder
        x = self.upconv4(bottleneck)  # 1024x16x16 --> 512x32x32
        x = torch.cat((x, x4), dim=1)  # Concatenate: (512x32x32 + 512x32x32) --> 1024x32x32
        x = self.dec_conv4(x)  # 1024x32x32 --> 512x32x32

        x = self.upconv3(x)  # 512x32x32 --> 256x64x64
        x = torch.cat((x, x3), dim=1)  # Concatenate: (256x64x64 + 256x64x64) --> 512x64x64
        x = self.dec_conv3(x)  # 512x64x64 --> 256x64x64

        x = self.upconv2(x)  # 256x64x64 --> 128x128x128
        x = torch.cat((x, x2), dim=1)  # Concatenate: (128x128x128 + 128x128x128) --> 256x128x128
        x = self.dec_conv2(x)  # 256x128x128 --> 128x128x128

        x = self.upconv1(x)  # 128x128x128 --> 64x256x256
        x = torch.cat((x, x1), dim=1)  # Concatenate: (64x256x256 + 64x256x256) --> 128x256x256
        x = self.dec_conv1(x)  # 128x256x256 --> 64x256x256

        # Final convolution
        x = self.conv_final(x)  # 64x256x256 --> 1x256x256
        return torch.sigmoid(x)  # Output: 1x256x256 (values between 0 and 1)

def display_images_masks(imagepath, maskpath):
    count = 0

    for image_file, mask_file in zip(image_files, mask_files):
        if count >= 4:  # Display only the first 4 pairs
            break

        # Full paths to the image and mask
        imagepath_full = os.path.join(imagepath, image_file)
        maskpath_full = os.path.join(maskpath, mask_file)

        # Load image and mask
        image = cv2.imread(imagepath_full)
        mask = cv2.imread(maskpath_full)

        # Validate image and mask loading
        if image is None:
            raise ValueError(f"Image at path {imagepath_full} could not be loaded")
        if mask is None:
            raise ValueError(f"Mask at path {maskpath_full} could not be loaded")

        plt.figure(figsize=(10, 10))

        # First subplot: Original Image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for display
        plt.title('Image')

        # Second subplot: Mask
        plt.subplot(1, 3, 2)
        plt.imshow(mask[:, :, 0], cmap='gray')  # Display the mask as grayscale
        plt.title('Mask')

        # Third subplot: Image with Contours
        plt.subplot(1, 3, 3)

        # Extract the mask for contours (assuming it's grayscale or in one channel)
        gray_mask = mask[:, :, 0] if len(mask.shape) == 3 else mask
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the original image
        sample_over_gt = cv2.drawContours(image.copy(), contours, -1, [0, 255, 0], thickness=2)
        plt.imshow(cv2.cvtColor(sample_over_gt, cv2.COLOR_BGR2RGB))
        plt.title('Image with Contours')

        plt.show()
        count += 1

display_images_masks(image_path, mask_path)

SIZE=256
CHANNEL=1
Num_Of_Classes=1

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),         # Resize the image to 256x256
    transforms.Grayscale(num_output_channels=1),  # Convert the image to grayscale
    transforms.ToTensor()                 # Convert the image to a tensor (automatically divides by 255)
])
if torch.cuda.is_available():
    device = torch.device("cuda", index=0)
else:
    device = torch.device("cpu", index=0)

X_train,X_test,y_train,y_test = train_test_split(image_files,mask_files,test_size = 0.2,random_state=42)

train_dataset = DatasetClass(
    image_files=X_train,
    mask_files=y_train,
    image_dir=image_path,
    mask_dir=mask_path,
    transform=transform_pipeline
)

test_dataset = DatasetClass(
    image_files=X_test,
    mask_files=y_test,
    image_dir=image_path,
    mask_dir=mask_path,
    transform=transform_pipeline
)

img,mask = train_dataset[0]


batch_size = 16
Train_DL= DataLoader(
    dataset = train_dataset,
    shuffle = True,
    batch_size = batch_size
)
Test_DL = DataLoader(
    dataset = test_dataset,
    shuffle = True,
    batch_size = batch_size
)

# Binary Cross-Entropy Loss
bce_loss = nn.BCELoss()

# Dice Loss function
def dice_loss(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def pixel_accuracy(pred, target):
    pred_bin = (pred > 0.5).float()  # Binarize predictions at 0.5 threshold
    correct = (pred_bin == target).sum()
    total = target.numel()  # Total number of pixels
    return correct.float() / total

# Helper function: Jaccard Index (IoU)
def jaccard_index(pred, target, smooth=1e-6):
    pred_bin = (pred > 0.5).float()  # Binarize predictions at 0.5 threshold
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Modified Combined Loss Function with Dice Score
def combined_loss(pred, target):
    bce = bce_loss(pred, target)
    dice = dice_loss(pred, target)
    total_loss = bce + dice
    dice_score = 1 - dice  # Dice score is complementary to Dice loss
    return total_loss, dice_score

num_epochs = 50
model = UNet()
model = model.to(device)
# Define the optimizer
learning_rate = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    track_loss = 0
    track_dice = 0  # Track Dice score across batches

    for i, (imgs, masks) in enumerate(dataloader):
        imgs = imgs.to(device)
        masks = masks.to(device)

        # Forward pass
        preds = model(imgs)
        loss, dice_score = loss_fn(preds, masks)  # Get combined loss and Dice score
        track_loss += loss.item()
        track_dice += dice_score.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Running metrics for display
        running_loss = round(track_loss / (i + 1), 4)
        running_dice = round((track_dice / (i + 1)) * 100, 2)

        # Display metrics every 100 batches
        if i % 100 == 0:
            print("Batch:", i + 1, "/", len(dataloader),
                  "Running Loss:", running_loss,
                  "Running Dice Score:", running_dice)

    # Calculate epoch metrics
    epoch_loss = running_loss
    epoch_dice = running_dice
    return epoch_loss, epoch_dice


def eval_one_epoch(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    track_loss = 0
    track_dice = 0
    num_batches = len(dataloader)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for i, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)
            preds = model(imgs)
            # Forward pass and compute combined loss and Dice score
            loss, dice_score = loss_fn(preds, masks)
            track_loss += loss.item()
            track_dice += dice_score.item()

    epoch_dice = (track_dice / num_batches) * 100  # Convert to percentage

    return epoch_dice


for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Training phase
    train_loss, train_dice = train_one_epoch(
        dataloader=Train_DL,
        model=model,
        loss_fn=combined_loss,
        optimizer=optimizer,
        device=device
    )

    # Validation phase
    val_dice = eval_one_epoch(
        dataloader=Test_DL,
        model=model,
        loss_fn=combined_loss,
        device=device
    )

# Helper function: Pixel Accuracy
def pixel_accuracy(pred, target):
    pred_bin = (pred > 0.5).float()  # Binarize predictions at 0.5 threshold
    correct = (pred_bin == target).sum()
    total = target.numel()  # Total number of pixels
    return correct.float() / total

# Helper function: Jaccard Index (IoU)
def jaccard_index(pred, target, smooth=1e-6):
    pred_bin = (pred > 0.5).float()  # Binarize predictions at 0.5 threshold
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# Updated final_eval function with additional metrics
def final_eval(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    track_loss = 0
    track_dice = 0
    track_pixel_accuracy = 0
    track_jaccard_index = 0
    num_batches = len(dataloader)
    num_displayed = 0  # Counter to limit visualizations to five images

    with torch.no_grad():  # Disable gradient computation for evaluation
        for i, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device)
            masks = masks.to(device)

            # Forward pass and compute combined loss and Dice score
            preds = model(imgs)
            loss, dice_score = loss_fn(preds, masks)
            track_loss += loss.item()
            track_dice += dice_score.item()

            # Compute pixel accuracy and Jaccard index
            acc = pixel_accuracy(preds, masks)
            iou = jaccard_index(preds, masks)
            track_pixel_accuracy += acc.item()
            track_jaccard_index += iou.item()

            # Display predictions for the first five images only
            if num_displayed < 5:
                preds_bin = (preds > 0.5).float()  # Binarize predictions at 0.5 threshold

                # Plot original image, ground truth mask, and predicted mask
                fig, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(imgs[0].cpu().squeeze(), cmap="gray")
                axs[0].set_title("Original Image (Transformed)")
                axs[0].axis("off")

                # Display the ground truth mask
                axs[1].imshow(masks[0].cpu().squeeze(), cmap="gray")
                axs[1].set_title("Ground Truth Mask")
                axs[1].axis("off")

                # Display the predicted mask
                axs[2].imshow(preds_bin[0].cpu().squeeze(), cmap="gray")
                axs[2].set_title("Predicted Mask")
                axs[2].axis("off")

                plt.show()
                num_displayed += 1
            else:
                break  # Increment the display counter

    # Calculate average metrics
    final_loss = track_loss / num_batches
    final_dice = (track_dice / num_batches) * 100  # Convert to percentage
    final_pixel_accuracy = (track_pixel_accuracy / num_batches) * 100  # Convert to percentage
    final_jaccard_index = (track_jaccard_index / num_batches) * 100  # Convert to percentage

    print(f"Final Loss: {final_loss:.4f}")
    print(f"Final Dice Score: {final_dice:.2f}%")
    print(f"Final Pixel Accuracy: {final_pixel_accuracy:.2f}%")
    print(f"Final Jaccard Index (IoU): {final_jaccard_index:.2f}%")

    return final_loss, final_dice, final_pixel_accuracy, final_jaccard_index

# Call the updated final_eval function
final_loss, final_dice, final_pixel_accuracy, final_jaccard_index = final_eval(Test_DL, model, combined_loss, device)

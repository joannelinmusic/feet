from sklearn.metrics import confusion_matrix
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix
import csv
import os
from PIL import Image
from torch.utils.data import Dataset

from path import my_path


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.classes = [c for c in os.listdir(root_dir) if c.isdigit()]
        self.image_paths = []
        self.labels = []
        self.image_names = []

        # Load images and labels
        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)
            if os.path.isdir(label_path):
                for image_name in os.listdir(label_path):
                    image_path = os.path.join(label_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(int(label))
                    self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        bw_image = Image.open(image_path)
        image = bw_image.convert("RGB")
        label = self.labels[idx]
        image_name = self.image_names[idx]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label, image_name
    

class MyNetwork(nn.Module):
    #def __init__(self, num_classes=2):
    def __init__(self): 
        super(MyNetwork, self).__init__()
        #self.flatten = nn.Flatten()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),            
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(53*53*16, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        #x = self.flatten(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
    

def preprocess(train, val):
    transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

    # Define the dataset
    train_set = CustomImageDataset(train, transform = transformations)
    val_set = CustomImageDataset(val, transform = transformations)

    # Define the batch size
    batch_size = 30
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size =batch_size, shuffle=True)

    return train_set, val_set, train_loader, val_loader

# def write_epoch_csv(epoch, prefix, y_true, y_pred, set):
#     file_path = os.path.join(directory, 'accuracy_predictions', f'{prefix}_epoch_{epoch}.csv')
#     with open(file_path, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Image', 'True Label', 'Predicted Label'])
#         for img, true_label, pred_label in zip(set.imgs, y_true, y_pred):
#             img_name = img[0][-16:]
#             writer.writerow([img_name, set.classes[true_label], set.classes[pred_label]])

def train(epochs):
    model = MyNetwork()
    device = "cpu"

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters())
    csv_directory = os.path.join(root_dir_864, 'accuracy_epoch_conf.csv')


    train_set, val_set, train_loader, val_loader = preprocess(train_folder, val_folder)

    train_loss = 0
    val_loss = 0
    train_accuracy = 0
    val_accuracy = 0
    y_pred_train = []
    y_true_train = []
    y_pred_val = []
    y_true_val = []
    confusion_matrices = []

    with open(csv_directory, 'w', newline='') as f:
        fieldNames = ['Epoch', 'Train_accuracy','Train_loss', 'Validation_accuracy', \
            'Validation_loss', 'Test_accuracy','Test_loss']
        myWriter = csv.DictWriter(f, fieldnames=fieldNames)
        myWriter.writeheader()

        for epoch in range(epochs):
            train_loss = 0
            val_loss = 0
            train_accuracy = 0
            val_accuracy = 0
            y_pred_train = []
            y_true_train = []
            y_name_train = []
            y_pred_val = []
            y_true_val = []
            y_name_val = []



            # Training the model
            model.train()
            counter = 0
            for inputs, labels, img_names in train_loader:
                
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model.forward(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()*inputs.size(0)
                
                ps = torch.exp(output)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                # Collect predictions and true labels for the training set
                y_pred_train.extend(top_class.squeeze().tolist())
                y_true_train.extend(labels.tolist())
                y_name_train.extend(img_names)
            
                # Print the progress
                counter += 1
                print(counter, "/", len(train_loader))
           
            # Evaluating the model
            model.eval()

            # Tell torch not to calculate gradients
            with torch.no_grad():
                counter = 0
                for inputs, labels, img_names in val_loader:
                    # print(inputs)
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    valloss = criterion(output, labels)
                    val_loss += valloss.item()*inputs.size(0)
                    
                    output = torch.exp(output)
                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Collect predictions and true labels for the validation set
                    y_pred_val.extend(top_class.squeeze().tolist())
                    y_true_val.extend(labels.tolist())
                    y_name_val.extend(img_names)
                    
                    counter += 1
                    
                    
            train_data = {
            'Image': [img_name for img_name in y_name_train],
            'True Label': [train_set.classes[true_label] for true_label in y_true_train],
            'Predicted Label': [train_set.classes[pred_label] for pred_label in y_pred_train]
            }
            train_df = pd.DataFrame(train_data)

            val_data = {
                'Image': [img_name for img_name in y_name_val],
                'True Label': [val_set.classes[true_label] for true_label in y_true_val],
                'Predicted Label': [val_set.classes[pred_label] for pred_label in y_pred_val]
            }
            val_df = pd.DataFrame(val_data)

            # Write DataFrames to CSV files
            train_csv_file = os.path.join(directory, 'accuracy_predictions', f'train_epoch_{epoch}.csv')
            val_csv_file = os.path.join(directory, 'accuracy_predictions', f'val_epoch_{epoch}.csv')

            train_df.to_csv(train_csv_file, index=False)
            val_df.to_csv(val_csv_file, index=False)

            # Get the average loss for the entire epoch
            train_loss = train_loss/len(train_loader.dataset)
            val_loss = val_loss/len(val_loader.dataset)

            
            print('Train Accuracy: ', train_accuracy / len(train_loader))
            print('Validation Accuracy: ', val_accuracy / len(val_loader))
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, val_loss))

            # Compute confusion matrix for training set
            train_matrix = confusion_matrix(y_true_train, y_pred_train)
            print("Confusion matrix for training set:")
            print(train_matrix)

            # Compute confusion matrix for validation set
            val_matrix = confusion_matrix(y_true_val, y_pred_val)
            print("Confusion matrix for validation set:")
            print(val_matrix)

            myWriter.writerow({'Epoch' : epoch+1, 'Train_accuracy' : train_accuracy/len(train_loader), \
                     'Train_loss' : train_loss, 'Validation_accuracy' : val_accuracy/len(val_loader), \
                     'Validation_loss' : val_loss})
            
            confusion_matrices.append({'Epoch': epoch, 'Train_matrix': train_matrix, 'Val_matrix': val_matrix})

    # Convert confusion matrices to pandas DataFrame
    df_confusion_matrices = pd.DataFrame(confusion_matrices)
    confusion_matrices_file_path = os.path.join(directory, 'accuracy_predictions', 'confusion_matrices_Q8.csv')
    df_confusion_matrices.to_csv(confusion_matrices_file_path, index=False)



def main():
    # preprocess()
    train(50)

if __name__ == '__main__':
    root_dir_864 = os.path.join(my_path, '864')
    train_folder = os.path.join(root_dir_864, 'train')
    val_folder = os.path.join(root_dir_864, 'validate')
    directory = os.path.join(my_path, 'csv_records_training')
    main()
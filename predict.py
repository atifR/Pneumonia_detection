# %% [markdown]
# ## Demo notebook to detect whether people are using safety gear or not
# 
# The data is taken from <br>
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data
# 
# 

# %% [markdown]
# imports ...

# %%
import os 
import torch
from torch import nn, optim
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as v2
import math
import time 

# %% [markdown]
# Parametes 

# %%
dataset_folder  = "Dataset"
train_folder = "train"
valid_folder = "val"
test_folder = "test"
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print (f"Settings ... \n dataset path \t\t {dataset_folder} \n batch_size \t\t {batch_size}" +
       f" \n device \t\t {device}")

# %% [markdown]
# Check the classes in the folder

# %%
class_labels = [name for name in os.listdir(f"{dataset_folder}/{train_folder}") if not name.startswith(".DS_Store")]
print(class_labels)
num_classes = len(class_labels)
print(num_classes)

# %% [markdown]
# Creating dataset from the image folder 

# %%
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transformations = v2.Compose([
    v2.ToImage(),
    v2.RandomHorizontalFlip(),
    v2.RandomVerticalFlip(),
    v2.RandomResizedCrop(size=(256, 256), antialias=True),
    v2.RandomRotation(degrees = (0, 170)),
    v2.ToDtype(dtype = torch.float32, scale = True),
    v2.Resize(size = (256, 256), antialias = True),
    v2.Normalize(mean, std)
])

train_dataset = ImageFolder(f"{dataset_folder}/{train_folder}", transform = transformations)
valid_dataset = ImageFolder(f"{dataset_folder}/{valid_folder}", transform = transformations)
test_dataset = ImageFolder(f"{dataset_folder}/{test_folder}", transform = transformations)

# %% [markdown]
# ## probe stats from the Image folder 

# %%
print (f"classes {train_dataset.classes}")
d = train_dataset.class_to_idx

num_classes =2

# %% [markdown]
# ## Data loader time 

# %%


train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset,shuffle=True,batch_size=batch_size)
test_dataloader = DataLoader(test_dataset,shuffle=True, batch_size=batch_size)

print(f"Lenght of training dataset {len(train_dataloader.dataset)}")
print(f"Lenght of validation dataset {len(valid_dataloader.dataset)}")
print(f"Lenght of test dataset {len(test_dataloader.dataset)}")

# %% [markdown]
# try ResNet 

# %%
res_net = torchvision.models.resnet50(weights='IMAGENET1K_V1')

for param in res_net.parameters():
    param.requires_grad = False

res_net.fc  = torch.nn.Sequential(
    nn.Linear(2048,128),
    nn.ReLU(),
    nn.Linear(
        in_features=128,
        out_features=num_classes
    ),
)



# %% [markdown]
# ANy other model(s) to evalaute can go here ... 

# %%


# %%
model_to_evaluate = res_net

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(res_net.fc.parameters())

# %% [markdown]
# training method 

# %%
def train_model(model,model_name,criterion,optimier, data_loader,device, num_epochs=0):
    model.to(device)
    for epoch in range(num_epochs):
        #loss_batches = 0
        loss_epoch=0;
        corrects_batches = 0
        count = 0
        start = time.time()
        correct = 0
        total = 0
        data_loading_begin = time.time()
        data_loading_time = 0
        data_processing_time = 0 
        for x,y in data_loader:
            
            x,y = x.to(device), y.to(device)
            data_loading_time += (time.time()-data_loading_begin)
            processing_time_begin = time.time()
            outputs = model(x)
            loss = criterion(outputs,y)
            optimier.zero_grad()
            loss.backward()
            optimier.step()
            data_processing_time += (time.time()-processing_time_begin)
            _,preds = torch.max(outputs,1)

            correct += (preds == y).sum().item()
            total += preds.size(0)
            loss_epoch += loss.item()
            count += 1
            data_loading_begin = time.time()
        #epoch_loss = loss_batches / len(data_loader)
        epoch_acc = correct / total
        print(f"\n epoch {epoch} Loss : {(loss_epoch/count):.4f} Accuracy {epoch_acc:.2f}, time : {time.time()-start} secs")
        print(f"data loading time {data_loading_time} secs, data processing time {data_processing_time} secs")
        if (epoch % 3 == 0 ):
          torch.save(model.state_dict(),f"{model_name}_{epoch:02d}_{epoch_acc:.2f}.h5")
    return model



# %%
train_model(model_to_evaluate,"res_net",criterion=criterion, optimier=optimizer,
            data_loader=train_dataloader, device=device, num_epochs=30)

torch.save(model_to_evaluate.state_dict(),'resNet.h5')

# %%
def calculate_accuracy(loader, model):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x,y = x.to(device), y.to(device)
            scores = model(x)
            _, predictions = scores.max(1)
            correct += (predictions == y).sum().item()
            total += predictions.size(0)
    model.train()
    return correct/total

# %%
print(f"Training Accuracy is {calculate_accuracy(train_dataloader, res_net)*100}")

print(f"Testing Accuracy is {calculate_accuracy(test_dataloader, res_net)*100}")

# %%




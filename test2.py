#making all the imports that we may need

import os
import random


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

#setting deivce for our calcunation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#calculating the test accuracy for the model-

test_loss = 0.00
correct_test = 0
total_test = 0
accuracy_test = 0.00
metrics=[]
labels=[]


def test_loop(test_loader,model,loss,metrics):

    model.eval()

    with torch.no_grad():
    #since we do not need to maintain records of gradients here

        for metric in metrics:
            metric.reset()
        
        for image, labels in test_loader:
            
            image = image.to(device=device)
            labels = labels.to(device=device)
            
            scores = model(image)
            _, predictions = scores.max(1)
            
            correct_test += (predictions == labels).sum()
            total_test += predictions.size(0)
        
        test_accuracy= correct_test/total_test
        test_accuracy
 

    model.train()


#finally,lets call the mf

test_loop(test_loader,model,loss,metrics)
    
    
    




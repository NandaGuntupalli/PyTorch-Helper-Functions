import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import random
import matplotlib.pyplot as plt

def train_segment(model, train_dataloader, loss_fn, optimizer, accuracy_fn, device):
  train_loss, train_acc = 0, 0
  for batch, (X, y) in enumerate(train_dataloader):
    model.train()
    X, y = X.to(device), y.to(device)
    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    
    train_loss += loss.item()
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"Train Loss: {train_loss/len(train_dataloader)} | Train Acc: {train_acc/len(train_dataloader)}")
  

def test(model, test_dataloader, loss_fn, accuracy_fn, device):
  test_loss, test_acc = 0, 0
  model.eval()
  with torch.inference_mode():
    for batch, (X, y) in enumerate(test_dataloader):
      X, y = X.to(device), y.to(device)
      y_pred = model(X)
      test_loss = loss_fn(y_pred, y)
      test_acc = accuracy_fn(y_pred.argmax(dim=1), y)

    print(f"Test Loss: {test_loss/len(test_dataloader)} | Test Acc: {test_acc/len(test_dataloader)}")


def simple_predictions(model, test_dataloader, device):
  with torch.no_grad():
      correct = 0
      total = 0
      for images, labels in test_dataloader:
          images = images.to(device)
          labels = labels.to(device)
          outputs = model_v3(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  
      print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))


def visual_predictions(model, data, class_names, device):
  pred_probs = []
  model.eval()
  with torch.inference_mode():
    for sample in data:
      sample = torch.unsqueeze(sample, dim=0).to(device)

      y_pred_logits = model(sample)
      y_pred_probs = torch.softmax(y_pred_logits, dim=1)
      pred_probs.append(y_pred_probs.cpu())

  make_preds = torch.stack(pred_probs, dim=0)

  random.seed(42)
  test_samples = []
  test_labels = []
  
  for sample, label in random.sample(list(data), k=9):
    test_samples.append(sample)
    test_labels.append(label)
  
  test_samples[0].shape
  
  pred_probabilities = make_preds
  pred_classes = pred_probabilities.argmax(dim=2)
  
  plt.figure(figsize=(9, 9))
  nrows = 3
  ncols = 3
  
  for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
  
    plt.imshow(sample.squeeze(), cmap="gray")
  
    pred_label = class_names[pred_classes[i].item()]
    true_label = class_names[test_labels[i]]
  
    title_text = f"Pred: {pred_label},\n True: {true_label}"
  
    if pred_label == true_label:
      plt.title(title_text, fontsize=10, c="g")
    else:
      plt.title(title_text, fontsize=10, c="r")
    plt.axis(False)

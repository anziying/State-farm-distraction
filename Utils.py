import torch
import numpy as np
import cv2
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pandas

def generate_img_tensor(folder, data_type):
    if data_type=='train':
        ###########################Training set#############################
        train_dataset_folder = folder
        train_imgs, train_labels = load_images_from_folder_10_class(train_dataset_folder)
        tensor_x_train = torch.Tensor(train_imgs) # transform to torch tensor
        tensor_y_train = torch.Tensor(train_labels)

        torch.save(tensor_x_train, 'training_image_tensor_7_8.pt')
        torch.save(tensor_y_train, 'training_label_tensor_7_8.pt')

    elif data_type=='test':
         ############################Testing set#############################
        test_dataset_folder = folder
        test_imgs, test_labels = load_images_from_folder(test_dataset_folder)
        tensor_x_test = torch.Tensor(test_imgs) # transform to torch tensor
        tensor_y_test = torch.Tensor(test_labels)

        torch.save(tensor_x_test, 'testing_image_tensor_7_8.pt')
        torch.save(tensor_y_test, 'testing_label_tensor_7_8.pt')
    else:
        raise ValueError(f'Not train and test')


def load_images_from_folder(folder):
    """
    :param folder: The folder that loads images
    :return: a list of ndarray-type images
    You can use this function to load all images in a certain folder into a list of np.ndarray
    """
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resize_img = cv2.resize(img, dsize=(299, 299), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            images.append(resize_img)
    return images


def load_images_from_folder_10_class(folder):
    """
    :param folder: The folder that loads images
    :return: image, labels
    Set "folder" as the path of training dataset (e.g. "distraction_data/imgs/train").
    There should be ten folders (c0-c9) under the path.
    The function returns images in a list of np.ndarray and a list of classification labels.
    """
    images = []
    labels = []
    class_list = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    for idx in range(len(class_list)):
        sub_folder = os.path.join(folder, class_list[idx])
        for filename in os.listdir(sub_folder):
            img = cv2.imread(os.path.join(sub_folder, filename))
            if img is not None:
                # resize the img for the network
                resize_img = cv2.resize(img, dsize=(299, 299), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
                img_frame = np.zeros((3, 299, 299))
                img_frame[0, :, :] = resize_img[:, :, 0]
                img_frame[1, :, :] = resize_img[:, :, 1]
                img_frame[2, :, :] = resize_img[:, :, 2]
                images.append(img_frame)
                labels.append(np.array([idx], dtype=np.longlong))
    return images, labels


def loadData(dataset, test_percentage, batch, num_thread=4):
    dataset_size = len(dataset)
    test_size = int(test_percentage * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset,
                                               [train_size, test_size])

    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch,
        num_workers=num_thread,
        shuffle=True)
    test_loader = DataLoader(
        test_dataset.dataset,
        batch_size=batch,
        num_workers=num_thread,
        shuffle=True)
    return train_loader, test_loader


def save_checkpoint(epoch, model, optimizer, fname='./Weights/checkpoint.pth'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, fname)


def train_loop(model, transform_fn, loss_fn, optimizer, dataloader, device, num_epochs, start_epoch=1):
    """
    :param model:
    :param transform_fn:
    :param loss_fn:
    :param optimizer:
    :param dataloader:
    :param num_epochs:
    :return:
    Use this function to train your model.
    """

    tbar = trange(start_epoch, num_epochs+1)
    for epoch in tbar:
        model.train()
        loss_total = 0.
        for i, (x, y) in enumerate(dataloader):
            x = transform_fn(x).to(device)
            pred = model(x)
            y = y.type(torch.LongTensor).to(device)
            loss = loss_fn(pred, y.squeeze(-1))
            # print(pred)
            # print(y.squeeze(-1))

            ## Parameter updates
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        tbar.set_description(f"Epoch: {epoch}, train loss: {(loss_total / len(dataloader)):.2e}")

        fname = './Weights/checkpoint_{}.pth'.format(model.name)
        save_checkpoint(epoch, model, optimizer, fname)

    return loss_total / len(dataloader)


def calculate_test_accuracy(model, transform_fn, dataloader, device):
    y_true = []
    y_pred = []
    tf = nn.Flatten()
    model.eval()
    for (xi, yi) in dataloader:
        with torch.no_grad():
            xi = transform_fn(xi).to(device)
            pred = model(xi)
            yi_pred = pred.argmax(-1).to(device)
        y_true.append(yi)
        y_pred.append(yi_pred)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    accuracy = (y_true.squeeze(-1) == y_pred).float().mean()
    return accuracy


def calculate_log_loss(model, dataloader, device):
    log_list = []
    model.eval()
    for (xi, yi) in dataloader:
        with torch.no_grad():
            pred = model(xi.to(device))
            pred_np = pred.cpu().detach().numpy()
            yi_np = yi.cpu().detach().numpy()
        logs = np.log(np.choose(yi_np.reshape(-1).astype(int), pred_np.T))
        log_list.append(np.mean(logs))
    return -np.mean(np.array(log_list))

from myXception import *
import torch
import numpy as np
import cv2
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torchvision import datasets
from torchvision.transforms import ToTensor
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas

rng_seed = 507
torch.manual_seed(rng_seed)


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
            images.append(img)
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
                img_frame = np.zeros((3, 480, 640))
                img_frame[0, :, :] = img[:, :, 0]
                img_frame[1, :, :] = img[:, :, 1]
                img_frame[2, :, :] = img[:, :, 2]
                images.append(img_frame)
                labels.append(np.array([idx], dtype=np.longlong))
    return images, labels


"""
Set your path of training dataset here
"""

# train_dataset_folder = r"distraction_data/imgs/train"
train_dataset_folder = r"small_dataset/imgs/train"

"""
Comment out the following snippet if you have your.pt files of images and labels in the root of directory. 
"""
############################Training set#############################

train_imgs, train_labels = load_images_from_folder_10_class(train_dataset_folder)
tensor_x_train = torch.Tensor(train_imgs) # transform to torch tensor
tensor_y_train = torch.Tensor(train_labels)

torch.save(tensor_x_train, 'training_image_tensor_7_8.pt')
torch.save(tensor_y_train, 'training_label_tensor_7_8.pt')

#############################Testing set#############################

# test_dataset_folder = r"small_test"
#
# test_imgs, test_labels = load_images_from_folder_10_class(test_dataset_folder)
# tensor_x_test = torch.Tensor(test_imgs) # transform to torch tensor
# tensor_y_test = torch.Tensor(test_labels)
#
# torch.save(tensor_x_test, 'testing_image_tensor_7_8.pt')
# torch.save(tensor_y_test, 'testing_label_tensor_7_8.pt')


"""
The following code loads the saved torch.Tensor file into a Dataloader
"""

tensor_x_train = torch.load('training_image_tensor_7_8.pt')
tensor_y_train = torch.load('training_label_tensor_7_8.pt')
# tensor_x_test = torch.load('testing_image_tensor_7_8.pt')
# tensor_y_test = torch.load('testing_label_tensor_7_8.pt')

train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset

def loadData(dataset, test_percentage, batch):
    dataset_size = len(dataset)
    test_size = int(test_percentage * dataset_size)
    train_size = dataset_size - test_size
    train_dataset, test_dataset = random_split(dataset,
                                               [train_size, test_size])
    train_loader = DataLoader(
        train_dataset.dataset,
        batch_size=batch,
        shuffle=True)
    test_loader = DataLoader(
        test_dataset.dataset,
        batch_size=batch,
        shuffle=True)
    return train_loader, test_loader



train_dataloader, test_dataloader = loadData(train_dataset, 0.3, 100)

# test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
# test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True)  # create your dataloader

def conv_out_size(slen, kernel_size, stride):
    """
    :param slen: Size length of the image. Should be an int.
    :param kernel_size: Int
    :param stride: Int
    :return: The size length of output after convolution
    This function considers 1-dim case.
    """
    return int((slen - kernel_size) / stride + 1)


def train_loop(model, transform_fn, loss_fn, optimizer, dataloader, num_epochs):
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
    tbar = tqdm(range(num_epochs))
    for _ in tbar:
        loss_total = 0.
        for i, (x, y) in enumerate(dataloader):
            x = transform_fn(x)
            pred = model(x)
            y = y.type(torch.LongTensor)
            loss = loss_fn(pred, y.squeeze(-1))
            # print(pred)
            # print(y.squeeze(-1))
            ## Parameter updates
            model.zero_grad()
            loss.backward()
            optimizer.step()

            loss_total += loss.item()
        tbar.set_description(f"Train loss: {loss_total / len(dataloader)}")

    return loss_total / len(dataloader)


def calculate_test_accuracy(model, transform_fn, dataloader):
    y_true = []
    y_pred = []
    tf = nn.Flatten()
    for (xi, yi) in dataloader:
        xi = transform_fn(xi)
        pred = model(xi)
        yi_pred = pred.argmax(-1)
        y_true.append(yi)
        y_pred.append(yi_pred)
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    accuracy = (y_true.squeeze(-1) == y_pred).float().mean()
    return accuracy


def calculate_log_loss(model, dataloader):
    log_list = []
    for (xi, yi) in dataloader:
        pred = model(xi)
        pred_np = pred.cpu().detach().numpy()
        yi_np = yi.cpu().detach().numpy()
        logs = np.choose(yi_np.reshape(-1).astype(int), pred_np.T)
        log_list.append(np.mean(logs))
    return np.mean(np.array(log_list))


convnet = Xception_Network(480, 640, num_classes=10)

convnet_optimizer = torch.optim.Adam(convnet.parameters(), lr=0.0002)


def s(x):
    return x


loss_functions = [torch.nn.CrossEntropyLoss(), nn.NLLLoss()]

train_loop(convnet, s, loss_functions[0], convnet_optimizer, train_dataloader, 10)

acc = calculate_test_accuracy(convnet, s, test_dataloader)

log_loss = calculate_log_loss(convnet, test_dataloader)

print(acc)
print(log_loss)

test_folder = r"distraction_data/imgs/test"


def predict_images_in_folder(folder, model):
    prediction_list = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_frame = np.zeros((3, 480, 640))
            img_frame[0, :, :] = img[:, :, 0]
            img_frame[1, :, :] = img[:, :, 1]
            img_frame[2, :, :] = img[:, :, 2]
            test_x = torch.Tensor(img_frame)
            pred_x = model(test_x)
            pred_x = pred_x.cpu().detach().numpy()
            pred_x = np.exp(pred_x)
            prediction_list.append((filename, pred_x[0].tolist()))
    prediction_list = sorted(prediction_list, key=lambda x: x[0])
    return prediction_list


result = predict_images_in_folder(test_folder, convnet)
result = [[x[0], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4], x[1][5], x[1][6], x[1][7], x[1][8], x[1][9]]
             for x in result]

result_df = pandas.DataFrame(result, columns=['img', 'c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
result_df.to_csv('submission.csv')

pass
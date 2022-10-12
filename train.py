from Models import Inception_Model, Xception_Model, Mixure_Model
from Utils import *

rng_seed = 507
torch.manual_seed(rng_seed)
print('Script Started')

"""
Set your path of training dataset here
"""

train_dataset_folder = r"distraction_data/imgs/train"
# train_dataset_folder = r"small_dataset/imgs/train"
test_dataset_folder = r"small_test"

# # todo: Comment out the following snippet if you have your.pt files of images and labels in the root of directory.
# generate_img_tensor(train_dataset_folder, data_type='train')
# generate_img_tensor(test_dataset_folder, data_type='test')


"""
The following code loads the saved torch.Tensor file into a Dataloader
"""
batch_size = 1
num_threads = 1

# load the data
tensor_x_train = torch.load('training_image_tensor_7_8.pt')
tensor_y_train = torch.load('training_label_tensor_7_8.pt')
# tensor_x_test = torch.load('testing_image_tensor_7_8.pt')
# tensor_y_test = torch.load('testing_label_tensor_7_8.pt')

train_dataset = TensorDataset(tensor_x_train, tensor_y_train)  # create your datset
train_dataloader, test_dataloader = loadData(train_dataset, 0.3, batch_size, num_threads)
# test_dataset = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
# test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)  # create your dataloader


'''
The training part
'''
Model_Name = 'Inception'
learn_rate = 0.001  # previously, lr=0.0002
num_epoch = 10
pretrained = None # input the fname of trained model like './Weights/checkpoint.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device.type)

print("Going to train...")

def trans(x):
    return x

def get_model(Model_name):
    if Model_name=='Inception':
        return Inception_Model
    elif Model_name=='Xception':
        return Xception_Model
    elif Model_name=='Mixture':
        return Mixure_Model
    else:
        raise ValueError("Unknown network type, please choose from 'Inception', 'Xception', and 'Mixture'!")

Model = get_model(Model_Name)
convnet = Model(299, 299, num_classes=10)
convnet_optimizer = torch.optim.Adam(convnet.parameters(), lr=learn_rate)
start_epoch = 1
if pretrained is not None:
    print('Load trained weight from {}'.format(pretrained))
    state = torch.load(pretrained)
    start_epoch = state['epoch']
    convnet.load_state_dict(state['state_dict'])
    convnet_optimizer.load_state_dict(state['optimizer'])

convnet = convnet.to(device)

loss_functions = [torch.nn.CrossEntropyLoss(), nn.NLLLoss()]
train_loop(convnet, trans, loss_functions[0], convnet_optimizer, train_dataloader, device, num_epoch, start_epoch)
acc = calculate_test_accuracy(convnet, trans, test_dataloader, device)
log_loss = calculate_log_loss(convnet, test_dataloader, device)

print(acc)
print(log_loss)


'''
The test part
'''
test_folder = r"distraction_data/imgs/test"
def predict_images_in_folder(folder, model):
    model.eval()
    prediction_list = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            resize_img = cv2.resize(img, dsize=(299, 299), fx=1, fy=1, interpolation=cv2.INTER_LINEAR)
            img_frame = np.zeros((3, 299, 299))
            img_frame[0, :, :] = resize_img[:, :, 0]
            img_frame[1, :, :] = resize_img[:, :, 1]
            img_frame[2, :, :] = resize_img[:, :, 2]

            with torch.no_grad():
                test_x = torch.Tensor([img_frame])
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
result_df.set_index('img')
result_df.to_csv('submission.csv')


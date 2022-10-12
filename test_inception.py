import torch
import os
import sys

from Inception import InceptionResnetV2


######################################################################
## Load parameters from HDF5 to Dict
######################################################################

def load_conv2d(state_dict, name_pth, name_tf):
    h5f = h5py.File('dump/InceptionResnetV2/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.conv.weight'] = torch.from_numpy(h5f['weights'][()]).permute(3, 2, 0, 1)
    out_planes = state_dict[name_pth + '.conv.weight'].size(0)
    state_dict[name_pth + '.bn.weight'] = torch.ones(out_planes)
    state_dict[name_pth + '.bn.bias'] = torch.from_numpy(h5f['beta'][()])
    state_dict[name_pth + '.bn.running_mean'] = torch.from_numpy(h5f['mean'][()])
    state_dict[name_pth + '.bn.running_var'] = torch.from_numpy(h5f['var'][()])
    h5f.close()


def load_conv2d_nobn(state_dict, name_pth, name_tf):
    h5f = h5py.File('dump/InceptionResnetV2/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weights'][()]).permute(3, 2, 0, 1)
    state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['biases'][()])
    h5f.close()


def load_linear(state_dict, name_pth, name_tf):
    h5f = h5py.File('dump/InceptionResnetV2/' + name_tf + '.h5', 'r')
    state_dict[name_pth + '.weight'] = torch.from_numpy(h5f['weights'][()]).t()
    state_dict[name_pth + '.bias'] = torch.from_numpy(h5f['biases'][()])
    h5f.close()


def load_mixed_5b(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0', name_tf + '/Branch_0/Conv2d_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_0b_5x5')
    load_conv2d(state_dict, name_pth + '.branch2.0', name_tf + '/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch2.1', name_tf + '/Branch_2/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth + '.branch2.2', name_tf + '/Branch_2/Conv2d_0c_3x3')
    load_conv2d(state_dict, name_pth + '.branch3.1', name_tf + '/Branch_3/Conv2d_0b_1x1')


def load_block35(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0', name_tf + '/Branch_0/Conv2d_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth + '.branch2.0', name_tf + '/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch2.1', name_tf + '/Branch_2/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth + '.branch2.2', name_tf + '/Branch_2/Conv2d_0c_3x3')
    load_conv2d_nobn(state_dict, name_pth + '.conv2d', name_tf + '/Conv2d_1x1')


def load_mixed_6a(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0', name_tf + '/Branch_0/Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth + '.branch1.2', name_tf + '/Branch_1/Conv2d_1a_3x3')


def load_block17(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0', name_tf + '/Branch_0/Conv2d_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_0b_1x7')
    load_conv2d(state_dict, name_pth + '.branch1.2', name_tf + '/Branch_1/Conv2d_0c_7x1')
    load_conv2d_nobn(state_dict, name_pth + '.conv2d', name_tf + '/Conv2d_1x1')


def load_mixed_7a(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0.0', name_tf + '/Branch_0/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch0.1', name_tf + '/Branch_0/Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth + '.branch2.0', name_tf + '/Branch_2/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch2.1', name_tf + '/Branch_2/Conv2d_0b_3x3')
    load_conv2d(state_dict, name_pth + '.branch2.2', name_tf + '/Branch_2/Conv2d_1a_3x3')


def load_block8(state_dict, name_pth, name_tf):
    load_conv2d(state_dict, name_pth + '.branch0', name_tf + '/Branch_0/Conv2d_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.0', name_tf + '/Branch_1/Conv2d_0a_1x1')
    load_conv2d(state_dict, name_pth + '.branch1.1', name_tf + '/Branch_1/Conv2d_0b_1x3')
    load_conv2d(state_dict, name_pth + '.branch1.2', name_tf + '/Branch_1/Conv2d_0c_3x1')
    load_conv2d_nobn(state_dict, name_pth + '.conv2d', name_tf + '/Conv2d_1x1')


def load():
    state_dict = {}
    load_conv2d(state_dict, name_pth='conv2d_1a', name_tf='Conv2d_1a_3x3')
    load_conv2d(state_dict, name_pth='conv2d_2a', name_tf='Conv2d_2a_3x3')
    load_conv2d(state_dict, name_pth='conv2d_2b', name_tf='Conv2d_2b_3x3')
    load_conv2d(state_dict, name_pth='conv2d_3b', name_tf='Conv2d_3b_1x1')
    load_conv2d(state_dict, name_pth='conv2d_4a', name_tf='Conv2d_4a_3x3')
    load_mixed_5b(state_dict, name_pth='mixed_5b', name_tf='Mixed_5b')

    for i in range(10):
        load_block35(state_dict, name_pth='repeat.' + str(i), name_tf='Repeat/block35_' + str(i + 1))

    load_mixed_6a(state_dict, name_pth='mixed_6a', name_tf='Mixed_6a')

    for i in range(20):
        load_block17(state_dict, name_pth='repeat_1.' + str(i), name_tf='Repeat_1/block17_' + str(i + 1))

    load_mixed_7a(state_dict, name_pth='mixed_7a', name_tf='Mixed_7a')

    for i in range(9):
        load_block8(state_dict, name_pth='repeat_2.' + str(i), name_tf='Repeat_2/block8_' + str(i + 1))

    load_block8(state_dict, name_pth='block8', name_tf='Block8')
    load_conv2d(state_dict, name_pth='conv2d_7b', name_tf='Conv2d_7b_1x1')
    load_linear(state_dict, name_pth='classif', name_tf='Logits')

    return state_dict


######################################################################
## Test
######################################################################

def test(model):
    from scipy import misc

    img = misc.imread('lena_299.png')
    inputs = torch.ones(1, 299, 299, 3)
    # inputs[0] = torch.from_numpy(img)

    inputs[0, 0, 0, 0] = -1
    inputs.transpose_(1, 3)
    inputs.transpose_(2, 3)

    print(inputs.mean())
    print(inputs.std())

    # inputs.sub_(0.5).div_(0.5)
    # inputs.sub_(inputs)
    # 1, 3, 299, 299

    outputs = model.forward(torch.autograd.Variable(inputs))
    h5f = h5py.File('dump/InceptionResnetV2/Logits.h5', 'r')
    outputs_tf = torch.from_numpy(h5f['out'][()])
    h5f.close()
    outputs = torch.nn.functional.softmax(outputs)
    print(outputs.sum())
    print(outputs[0])
    print(outputs_tf.sum())
    print(outputs_tf[0])
    print(torch.dist(outputs.data, outputs_tf))
    return outputs


def test_conv2d(module, name):
    # global output_tf
    h5f = h5py.File('dump/InceptionResnetV2/' + name + '.h5', 'r')
    output_tf_conv = torch.from_numpy(h5f['conv_out'][()])
    output_tf_conv.transpose_(1, 3)
    output_tf_conv.transpose_(2, 3)
    output_tf_relu = torch.from_numpy(h5f['relu_out'][()])
    output_tf_relu.transpose_(1, 3)
    output_tf_relu.transpose_(2, 3)
    h5f.close()

    def test_dist_conv(self, input, output):
        print(name, 'conv', torch.dist(output.data, output_tf_conv))
        module.conv.register_forward_hook(test_dist_conv)

    def test_dist_relu(self, input, output):
        print(name, 'relu', torch.dist(output.data, output_tf_relu))
        module.relu.register_forward_hook(test_dist_relu)


def test_conv2d_nobn(module, name):
    # global output_tf
    h5f = h5py.File('dump/InceptionResnetV2/' + name + '.h5', 'r')
    output_tf = torch.from_numpy(h5f['conv_out'][()])
    output_tf.transpose_(1, 3)
    output_tf.transpose_(2, 3)
    h5f.close()

    def test_dist(self, input, output):
        print(name, 'conv+bias', torch.dist(output.data, output_tf))

    module.register_forward_hook(test_dist)


def test_mixed_5b(module, name):
    test_conv2d(module.branch0, name + '/Branch_0/Conv2d_1x1')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_0b_5x5')
    test_conv2d(module.branch2[0], name + '/Branch_2/Conv2d_0a_1x1')
    test_conv2d(module.branch2[1], name + '/Branch_2/Conv2d_0b_3x3')
    test_conv2d(module.branch2[2], name + '/Branch_2/Conv2d_0c_3x3')
    test_conv2d(module.branch3[1], name + '/Branch_3/Conv2d_0b_1x1')


def test_block35(module, name):
    test_conv2d(module.branch0, name + '/Branch_0/Conv2d_1x1')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_0b_3x3')
    test_conv2d(module.branch2[0], name + '/Branch_2/Conv2d_0a_1x1')
    test_conv2d(module.branch2[1], name + '/Branch_2/Conv2d_0b_3x3')
    test_conv2d(module.branch2[2], name + '/Branch_2/Conv2d_0c_3x3')
    test_conv2d_nobn(module.conv2d, name + '/Conv2d_1x1')


def test_mixed_6a(module, name):
    test_conv2d(module.branch0, name + '/Branch_0/Conv2d_1a_3x3')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_0b_3x3')
    test_conv2d(module.branch1[2], name + '/Branch_1/Conv2d_1a_3x3')


def test_block17(module, name):
    test_conv2d(module.branch0, name + '/Branch_0/Conv2d_1x1')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_0b_1x7')
    test_conv2d(module.branch1[2], name + '/Branch_1/Conv2d_0c_7x1')
    test_conv2d_nobn(module.conv2d, name + '/Conv2d_1x1')


def test_mixed_7a(module, name):
    test_conv2d(module.branch0[0], name + '/Branch_0/Conv2d_0a_1x1')
    test_conv2d(module.branch0[1], name + '/Branch_0/Conv2d_1a_3x3')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_1a_3x3')
    test_conv2d(module.branch2[0], name + '/Branch_2/Conv2d_0a_1x1')
    test_conv2d(module.branch2[1], name + '/Branch_2/Conv2d_0b_3x3')
    test_conv2d(module.branch2[2], name + '/Branch_2/Conv2d_1a_3x3')


def test_block8(module, name):
    test_conv2d(module.branch0, name + '/Branch_0/Conv2d_1x1')
    test_conv2d(module.branch1[0], name + '/Branch_1/Conv2d_0a_1x1')
    test_conv2d(module.branch1[1], name + '/Branch_1/Conv2d_0b_1x3')
    test_conv2d(module.branch1[2], name + '/Branch_1/Conv2d_0c_3x1')
    test_conv2d_nobn(module.conv2d, name + '/Conv2d_1x1')

######################################################################
## Main
######################################################################

if __name__ == "__main__":
    import h5py

    model = InceptionResnetV2()
    state_dict = load()
    model.load_state_dict(state_dict)
    model.eval()

    os.system('mkdir -p save')
    torch.save(model, 'save/inceptionresnetv2.pth')
    torch.save(state_dict, 'save/inceptionresnetv2_state.pth')

    test_conv2d(model.conv2d_1a, 'Conv2d_1a_3x3')
    test_conv2d(model.conv2d_2a, 'Conv2d_2a_3x3')
    test_conv2d(model.conv2d_2b, 'Conv2d_2b_3x3')
    test_conv2d(model.conv2d_3b, 'Conv2d_3b_1x1')
    test_conv2d(model.conv2d_4a, 'Conv2d_4a_3x3')

    test_mixed_5b(model.mixed_5b, 'Mixed_5b')

    for i in range(len(model.repeat._modules)):
        test_block35(model.repeat[i], 'Repeat/block35_' + str(i + 1))

    test_mixed_6a(model.mixed_6a, 'Mixed_6a')

    for i in range(len(model.repeat_1._modules)):
        test_block17(model.repeat_1[i], 'Repeat_1/block17_' + str(i + 1))

    test_mixed_7a(model.mixed_7a, 'Mixed_7a')

    for i in range(len(model.repeat_2._modules)):
        test_block8(model.repeat_2[i], 'Repeat_2/block8_' + str(i + 1))

    test_block8(model.block8, 'Block8')

    test_conv2d(model.conv2d_7b, 'Conv2d_7b_1x1')

    outputs = test(model)
    # test_conv2d(model.features[1], 'Conv2d_2a_3x3')
    # test_conv2d(model.features[2], 'Conv2d_2b_3x3')
    # test_conv2d(model.features[3].conv, 'Mixed_3a/Branch_1/Conv2d_0a_3x3')
    # test_mixed_4a_7a(model.features[4], 'Mixed_4a')

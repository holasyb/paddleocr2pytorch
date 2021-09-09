import torch
import os, sys
from collections import OrderedDict
import numpy as np
import cv2
import torch
# from crnn_mbv3 import CrnnModel
from lcnet_backbone import CrnnModel

def read_paddle_weights(weights_path):
    if 'infer' in weights_path:
        import paddle
        para_state_dict = paddle.load(weights_path)
        print(type(para_state_dict))
        print(para_state_dict.keys())
        opti_state_dict = ''
    else:
        import paddle.fluid as fluid
        with fluid.dygraph.guard():
            para_state_dict, opti_state_dict = fluid.load_dygraph(weights_path)
        print(para_state_dict.keys())
    return para_state_dict, opti_state_dict


def rename_lstm(instr):
    # print('lstm', instr)
    if 'reverse' in instr:
        back = 'b'
    else:
        back = 'f'
    instr = instr.strip('_reverse').strip('neck.encoder.lstm.')
    bias = instr.split('l')[0][:-1]
    # print('lstm', instr)
    r_name = 'neck.encoder.lstm.' + instr[-1] + '.cell_' + back + 'w.' + bias
    return r_name

def load_paddle_weights(torch_net, paddle_weights,path):
    para_state_dict, opti_state_dict = paddle_weights
    # print(para_state_dict.keys())
    [print('paddle', i, para_state_dict[i].shape) for i in para_state_dict.keys()]# if 'fc' in i]
    # print(para_state_dict[tmp[0]].shape)
    # assert 'neck.encoder.lstm.weight_ih_l0' in para_state_dict.keys()

    for k,v in torch_net.state_dict().items():
        # print('torch', k, v.shape)
        keyword = 'block_list.'
        if keyword in k:
            # replace: block_list.
            name = k
            # name = k.replace(keyword, '')
        else:
            name = k
        fc_num = 1
        if name.endswith('num_batches_tracked'):
            continue
        if '.se.' in name:
            name = name.replace('.se.', '.mid_se.')
        if '.bn.' in name:
            name = name.replace('.bn.', '._batch_norm.')
        if name.endswith('running_mean'):
            ppname = name.replace('running_mean', '_mean')
        elif name.endswith('running_var'):
            ppname = name.replace('running_var', '_variance')
        elif name.endswith('bias') or name.endswith('weight'):
            ppname = name.replace('.conv.', '._conv.')
        elif 'lstm' in name:
            ppname = rename_lstm(name)

        else:
            print('Redundance:')
            print(name)
            raise ValueError
        # print(ppname)
        print(k, v.shape, ppname, para_state_dict[ppname].shape)
        try:
            if ppname.endswith('fc1.weight') or ppname.endswith('fc2.weight'):
                torch_net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname].T))
            else:
                torch_net.state_dict()[k].copy_(torch.Tensor(para_state_dict[ppname]))
        except Exception as e:
            print('pytorch: {}, {}'.format(k, v.size()))
            print('paddle: {}, {}'.format(ppname, para_state_dict[ppname].shape))
            raise e
    state_dict = {'state_dict': torch_net.state_dict()}
    torch.save(state_dict, os.path.split(os.path.split(path)[0])[-1] + '.pth', _use_new_zipfile_serialization=False)
    print('model is loaded.')
    return torch_net

if __name__ == '__main__':
    model_path = '/home/yubosun/桌面/code/ocr/sbcocr/PaddleOCR2Pytorch-main/source/ch_PP-OCRv2_rec_infer'
    model_path0 = [i for i in os.listdir(model_path) if i.endswith('params')][0].split('.')[0]
    model_path = os.path.join(model_path, model_path0)
    para_state_dict, opti_state_dict = read_paddle_weights(model_path)
    torch_crnn = CrnnModel(n_classes=6625)
    torch_net = load_paddle_weights(torch_crnn, (para_state_dict, opti_state_dict), model_path)
    print('done')
    image = cv2.imread('./doc/imgs_words/ch/word_1.jpg')
    image = cv2.resize(image, (320, 32))
    mean = 0.5
    std = 0.5
    scale = 1. / 255
    norm_img = (image * scale - mean) / std
    transpose_img = norm_img.transpose(2, 0, 1)
    transpose_img = np.expand_dims(transpose_img, 0).astype(np.float32)
    inp = torch.Tensor(transpose_img)
    print('inp:', np.sum(transpose_img), np.mean(transpose_img), np.max(transpose_img), np.min(transpose_img))

    out = torch_net(inp)
    out = out.data.numpy()
    print('out:', np.sum(out), np.mean(out), np.max(out), np.min(out))

    # save
    # converter.save_pytorch_weights('ch_ptocr_mobile_v2.0_rec_infer.pth')
    # converter.save_pytorch_weights('ch_ptocr_mobile_v2.0_rec_pre.pth')
    print('done.')
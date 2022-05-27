import os
import sys
import csv
from PIL import Image
from collections import OrderedDict
import torch
from torchvision import transforms

module_path = os.path.abspath(os.path.join('/home/lingjia/Documents/diamond/EfficientNet'))
if module_path not in sys.path:
    sys.path.append(module_path)
from efficientnet_pytorch import EfficientNet


if __name__ == '__main__':

    # class_list = {'Pinpoint': 1, 'Crystal': 2, 'Needle': 3, 'Feather': 4, 'Internal_graining': 5,
    #                        'Cloud': 6, 'Twinning_wisp': 7, 'Nick': 8, 'Pit': 9, 'Burn_mark': 10}
    class_list = {'Cloud': 1, 'Crystal': 2, 'Feather': 3, 'Twinning_wisp': 4}
    labels_map = list(class_list.keys())

    # Choose model
    # labels_map = oao_groups['6']
    print(f'Infer based on OAO model multiclass once')

    # Load checkpoint
    model_name = 'efficientnet-b0'
    image_size = EfficientNet.get_image_size(model_name) # 224
    model = EfficientNet.from_name(model_name, num_classes=4)

    model_path = '/home/lingjia/Documents/diamond_result/cls_multi-class_EfficientNet/oao_strategy'
    checkpoint_path = os.path.join(model_path,'multiclass_once','model_best.pth')
    # checkpoint = torch.load(checkpoint_path)
    # model.load_state_dict(checkpoint['state_dict'])

    state_dict = torch.load(checkpoint_path)['state_dict']
    # create new OrderedDict that does not contain 'module.'
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove 'module.'
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)


    # Infer set
    testList = {}
    data_path = '/media/hdd/diamond_data/cls_multi-class_EfficientNet'
    for cls_name in class_list.keys():
        # if not cls_name.lower() == 'pinpoint':
        with open(os.path.join(data_path,cls_name,'test_ids.txt'), 'r') as file:
            tmpList = file.readlines()
        tmpList = [n.split('\n')[0] for n in tmpList]
        for tmp in tmpList:
            testList[tmp] = [tmp,class_list[cls_name],cls_name]


    infer_info = [['img_name','cls_name','cls_id',f'prob_{labels_map[0]}',f'prob_{labels_map[1]}',f'prob_{labels_map[2]}',f'prob_{labels_map[3]}']]
    num_idx = len(testList.keys())
    for idx, img_n in enumerate(testList.keys()):

        # Open image
        cls_name = testList[img_n][2]
        img = Image.open(os.path.join(data_path,cls_name,img_n))

        # Preprocess image
        tfms = transforms.Compose([transforms.Resize((image_size,image_size)), 
                                transforms.CenterCrop(image_size), 
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
        img = tfms(img).unsqueeze(0)

        # Classify with EfficientNet
        model.eval()
        with torch.no_grad():
            logits = model(img)
 
        probs = torch.softmax(logits, dim=1)
        tmp_info = [img_n,cls_name,class_list[cls_name],probs[0,0].item(),probs[0,1].item(),probs[0,2].item(),probs[0,3].item()]
        print(f'{idx}/{num_idx} {img_n} GT:{cls_name} Pred:{probs}')

        infer_info.append(tmp_info)


    save_path = '/media/hdd/diamond_infer/cls_multi-class_EfficientNet_0406'
    os.makedirs(save_path,exist_ok=True)
    infer_info_file = os.path.join(save_path,'infer_on_multiclass.csv')
    with open(infer_info_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(infer_info)
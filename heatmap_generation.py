import os
import PIL
import cv2
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torchvision.models as models
from torchvision.utils import make_grid, save_image
from tvision import *
from gradcam.utils import visualize_cam, Normalize
from gradcam.gradcam import GradCAM, GradCAMpp
from PIL import Image, ImageDraw
from resnet import *
import json
import pandas as pd
from BBOXES_from_GRADCAM import BBoxerwGradCAM # load class from .py file

def get_IoU(truth_coords, pred_coords):
    pred_area = pred_coords[2]*pred_coords[3]
    truth_area = truth_coords[2]*truth_coords[3]
    # coords of intersection rectangle
    x1 = max(truth_coords[0], pred_coords[0])
    y1 = max(truth_coords[1], pred_coords[1])
    x2 = min(truth_coords[2], pred_coords[2])
    y2 = min(truth_coords[3], pred_coords[3])
    # area of intersection rectangle
    interArea = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    # area of prediction and truth rectangles
    boxTruthArea = (truth_coords[2] - truth_coords[0] + 1) * (truth_coords[3] - truth_coords[1] + 1)
    boxPredArea = (pred_coords[2] - pred_coords[0] + 1) * (pred_coords[3] - pred_coords[1] + 1)
    # intersection over union 
    iou = interArea / float(boxTruthArea + boxPredArea - interArea)
    return iou

pcam_model = ResidualNet('ImageNet', 50, 8, 'TripletAttention')
pcam_model.load_state_dict(torch.load("./checkpoint/model_BCE_837.pth"))
pcam_model.cuda()

cam_dict = dict()
pcam_resnet_model_dict = dict(type='resnet', arch=pcam_model, layer_name='layer4', input_size=(224, 224))
pcam_resnet_gradcam = GradCAM(pcam_resnet_model_dict, True)
pcam_resnet_gradcampp = GradCAMpp(pcam_resnet_model_dict, True)
cam_dict['pcam_resnet'] = [pcam_resnet_gradcam, pcam_resnet_gradcampp]

with open('gt_dict.json', 'r') as fp:
    data = json.load(fp)
gt_images = list(data.keys())
fopen = open("bb_results.csv", "w")
thresold = 0.65
gt = pd.read_csv('./ground_truth/BB.csv')
CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltrate', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax']
for c in CLASS_NAMES:
    disease_data = gt[gt["Finding Label"] == c]
    acc = 0
    _image_name = ""
    
    for img in list(disease_data["Image Index"]):
        try:
            img_name = "./ground_truth/" + img
            x, y, w, h = data[img][0], data[img][1], data[img][2], data[img][3]

            #print("Processing Image {} : {}, {}, {}, {}".format(img_name, x, y, w, h))

            pil_img = PIL.Image.open(img_name).convert('RGB')
            pil_ref = PIL.Image.open(img_name).convert('RGB')

            shape = [x, y, x + w, y + h]
            reference = ImageDraw.Draw(pil_ref)  
            reference.rectangle(shape,  outline ="yellow", width = 4)

            gt_image = pil_ref.resize((224,224), Image.ANTIALIAS)
            gt_image.save("./output/gtimage_"+img_name.split("/")[-1])


            normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            torch_img = torch.from_numpy(np.asarray(pil_img)).permute(2, 0, 1).unsqueeze(0).float().div(255)
            torch_img = F.upsample(torch_img, size=(224, 224), mode='bilinear', align_corners=False)
            normed_torch_img = normalizer(torch_img)

            torch_img_ref = torch.from_numpy(np.asarray(pil_ref)).permute(2, 0, 1).unsqueeze(0).float().div(255)
            torch_img_ref = F.upsample(torch_img_ref, size=(224, 224), mode='bilinear', align_corners=False)



            images = []
            image_resizing_scale = [1024, 1024]
            bbox_scaling = [1,1,1,1] 
            for gradcam, gradcam_pp in cam_dict.values():
                mask, _ = gradcam(normed_torch_img.cuda())
                heatmap, result = visualize_cam(mask.cpu(), torch_img.cpu())

                mask_pp, _ = gradcam_pp(normed_torch_img.cuda())
                heatmap_pp, result_pp = visualize_cam(mask_pp.cpu(), torch_img.cpu())

                upscaled_heatmap = F.upsample(heatmap_pp.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
                hmap = upscaled_heatmap.squeeze(0)[0]
                hmap = np.where(hmap >= 0.95, hmap, 0)
                bbox = BBoxerwGradCAM(None,
                                  hmap,
                                  img_name,
                                  image_resizing_scale,
                                  bbox_scaling)
                rect_coords, polygon_coords = bbox.get_bboxes()
                #print(rect_coords)
                save_image(torch.from_numpy(hmap), "./output/mask"+img_name.split("/")[-1])

                #images.append(torch.stack([torch_img.squeeze().cpu(), torch_img_ref.squeeze().cpu(), result, result_pp, heatmap_pp], 0))
                images.append(torch.stack([torch_img.squeeze().cpu()], 0))
            images = make_grid(torch.cat(images, 0), nrow=5)
            save_image(images, "./output/original_"+img_name.split("/")[-1])
            save_image(result_pp, "./output/pred_"+img_name.split("/")[-1])

            pil_img = PIL.Image.open("./output/pred_"+img_name.split("/")[-1]).convert('RGB')
            img1 = ImageDraw.Draw(pil_img)  
#             img1.rectangle(shape,  outline ="red", width = 2)
            max_iou = 0.0

            for i in range(0, len(rect_coords)):
                rect_cord = rect_coords[i]
                #img1.rectangle(rect_cord,  outline ="red", width = 2)
                iou = get_IoU(shape, rect_cord)
                max_iou = max(max_iou, iou)
            pil_img.save("./output/pred_"+img_name.split("/")[-1])
            fopen.write("{},{},{}\n".format(img, c, max_iou))
            if max_iou >= thresold:
                acc += 1
                _image_name = img
        except:
            print("Error")
            continue
    print("Category : {} Threshold : {} Accuracy : {}".format(c, thresold, acc/len(disease_data)))
    print("Best Image : {}".format(_image_name))
    
    
# print("Final Accuracy : {} {}".format(thresold + 0.05, acc/(end- start + 1)))
# print("Best Image for Display: {}".format(_image_name))



#         x = torchvision.transforms.ToPILImage()(upscaled_heatmap.squeeze(0))
#         x = torchvision.transforms.Grayscale()(x)
#         x = torchvision.transforms.ToTensor()(x)
#         hmap = x.squeeze(0)
#         print(hmap)
#         print(hmap.shape)
#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
from cv2 import imshow as cv2_imshow
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
class BBoxerwGradCAM():
    
    def __init__(self,learner,heatmap,image_path,resize_scale_list,bbox_scale_list):
        self.learner = learner
        self.heatmap = heatmap
        self.image_path = image_path
        self.resize_list = resize_scale_list
        self.scale_list = bbox_scale_list
        
        self.og_img, self.smooth_heatmap = self.heatmap_smoothing()
        
        self.bbox_coords, self.poly_coords, self.grey_img, self.contours = self.form_bboxes()
        
    def heatmap_smoothing(self):
        og_img = cv2.imread(self.image_path)
        heatmap = cv2.resize(self.heatmap, (self.resize_list[0],self.resize_list[1])) # Resizing
        og_img = cv2.resize(og_img, (self.resize_list[0],self.resize_list[1])) # Resizing
        '''
        The minimum pixel value will be mapped to the minimum output value (alpha - 0)
        The maximum pixel value will be mapped to the maximum output value (beta - 155)
        Linear scaling is applied to everything in between.
        These values were chosen with trial and error using COLORMAP_JET to deliver the best pixel saturation for forming contours.
        '''
        heatmapshow = cv2.normalize(heatmap, None, alpha=0, beta=155, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
        
        return og_img, heatmapshow
    
    def show_smoothheatmap(self):
        cv2_imshow(self.smooth_heatmap)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_bboxrectangle(self):
        cv2.rectangle(self.og_img,
                      (self.bbox_coords[0],self.bbox_coords[1]),
                      (self.bbox_coords[0]+self.bbox_coords[2],self.bbox_coords[1]+self.bbox_coords[3]),
                      (0,0,0),3)
        cv2_imshow(self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def show_contouredheatmap(self):
        img_col = cv2.merge([self.grey_img,self.grey_img,self.grey_img]) # merge channels to create color image (3 channels)
        cv2.fillPoly(img_col, self.contours, [36,255,12]) # fill contours on 3 channel image
        cv2_imshow(img_col)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def show_bboxpolygon(self):
        cv2.polylines(self.og_img,self.poly_coords,True,(0,0,0),2)
        cv2_imshow(self.og_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def merge_box(self, box1, box2):
        x1, y1, x2, y2 = min(box1[0], box2[0]), min(box1[1], box2[1]), max(box1[2], box2[2]), max(box1[3], box2[3])
        return [x1, y1, x2, y2]
    
    def form_bboxes(self):
        grey_img = cv2.cvtColor(self.smooth_heatmap, cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(grey_img,127,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh, 1, 2)
        list_BB = []
        for item in range(len(contours)):
            cnt = contours[item]
            if len(cnt) > 20:
                #print(len(cnt))
                x,y,w,h = cv2.boundingRect(cnt) # x, y is the top left corner, and w, h are the width and height respectively
                poly_coords = [cnt] # polygon coordinates are based on contours
                
                x = int(x*self.scale_list[0]) # rescaling the boundary box based on user input
                y = int(y*self.scale_list[1])
                w = int(w*self.scale_list[2])
                h = int(h*self.scale_list[3])
                
                list_BB.append([x, y , x + w, y + h])
            
            else: print("contour error (too small)")
        from itertools import combinations
        for item in list(combinations(np.arange(0, len(list_BB)), 2)):
            list_BB.append(self.merge_box(list_BB[item[0]], list_BB[item[1]]))
        for item in list(combinations(np.arange(0, len(list_BB)), 2)):
            list_BB.append(self.merge_box(list_BB[item[0]], list_BB[item[1]]))
#         print(list_BB)
#         import sys
#         sys.exit(0)
#         bb_list = pd.DataFrame(list_BB)
#         x1, y1, x2, y2 = min(bb_list[0]), min(bb_list[1]), max(bb_list[2]), max(bb_list[3])
        return list_BB, poly_coords, grey_img, contours     
    def get_bboxes(self):
        return self.bbox_coords, self.poly_coords
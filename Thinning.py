#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:36:34 2018

@author: dohoonkim
"""

from os.path import normpath as fn
import numpy as np
import cv2
import queue
from operator import add
from collections import deque
#from pythonds.basic.stack import Stack

#img = np.float32(imread(fn('easy/24708.1_1 at 20X.jpg')))
def crop(img,pts):
    bounding_rect = cv2.boundingRect(pts)
    x,y,w,h = bounding_rect
    cropped = img[y:y+h, x:x+w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(cropped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(cropped, cropped, mask=mask)
    bg = np.ones_like(cropped, np.uint8)*255
    cv2.bitwise_not(bg,bg, mask=mask)
    dst = bg+ dst
    return dst

#need to change brightness for dark image
def adaptivecontrast(img,k):
    tempimg = img.copy()
    contrastval = 50
    for i in range(tempimg.shape[0]):
        for j in range(tempimg.shape[1]):
            isum = 0
            count = 0
            for ki in range(-k,k+1):
                for kj in range(-k,k+1):
                    if ki+i < tempimg.shape[0] and ki+i >= 0 and kj+j < tempimg.shape[1] and kj+j >= 0:
                        if img[ki+i][kj+j] != 0:
                            isum += img[ki+i][kj+j]
                            count += 1
            if count > 0:
                if img[i][j] * count > isum:
                    if img[i][j] + contrastval > 255:
                        tempimg[i][j] = 255
                    else:
                        tempimg[i][j] = img[i][j] + contrastval
            else:
                tempimg[i][j] = 0
    return tempimg 

def thresholding(img,k):
    bin_img = np.zeros((img.shape),dtype=int)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ksum = 0
            count = 0
            if img[i][j] != 0:
                for ki in range(-k,k+1):
                    for kj in range(-k,k+1):
                        if ki+i < img.shape[0] and ki+i >= 0 and kj+j < img.shape[1] and kj+j >= 0:
                            if img[ki+i][kj+j] != 0:
                                ksum += img[ki+i][kj+j]
                                count += 1
                if img[i][j]*count > ksum:
                    bin_img[i][j] = 1
            
    return bin_img

def numcomponents(img):
    labeledimg = np.zeros((img.shape),dtype=int)
    vflag = np.zeros((img.shape),dtype=int)
    label = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if vflag[i][j] == 1:
                continue
            label += 1
            queue_points = deque([])
            queue_points.append([i,j])
            while not len(queue_points)==0:
                [x,y] = queue_points.popleft()
                if x < 0 or y < 0 or x >= img.shape[0] or y >= img.shape[1]:
                    continue
                if vflag[x][y] == 1:
                    continue
                vflag[x][y] = 1
                if img[x][y] == 0:
                    continue
                labeledimg[x][y] = label
                queue_points.append([x-1,y])
                queue_points.append([x+1,y])
                queue_points.append([x,y-1])
                queue_points.append([x,y+1])
                queue_points.append([x-1,y-1])
                queue_points.append([x-1,y+1])
                queue_points.append([x+1,y-1])
                queue_points.append([x+1,y+1])
    return label,labeledimg

def getlargest(k,num_components,label_img,bin_img):
    counts = np.zeros((num_components),dtype=int)
    for i in range(bin_img.shape[0]):
        for j in range(bin_img.shape[1]):
            counts[label_img[i][j]] += 1
    counts = counts.tolist()
    whilecount = 0
    while whilecount < k:
        maxi = 1
        maxs = counts[maxi]
        for i in range(2,num_components):
            if counts[i] > maxs:
                maxi = i
                maxs = counts[i]
        for x in range(bin_img.shape[0]):
            for y in range(bin_img.shape[1]):             
                if label_img[x][y] == maxi:
                    bin_img[x][y] = 1
        whilecount += 1
        counts[maxi] = 0
    return counts,bin_img

def erode(bin_img):
    structcross = [[0,1],[0,-1],[1,0],[-1,0]]
    structsquare = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    temp = bin_img.copy()
    output = bin_img.copy()
    for i in range(bin_img.shape[0]-1,-1,-1):
        for j in range(0,bin_img.shape[1]):
            for s in structcross:
                if i + s[0] >= 0 and i + s[0] < bin_img.shape[0] and j + s[1] >= 0 and j + s[1] < bin_img.shape[1]:
                    if temp[i+s[0]][j+s[1]] == 0:
                        output[i][j] = 0
    return output

def dilate(bin_img):
    structcross = [[0,1],[0,-1],[1,0],[-1,0]]
    structsquare = [[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1]]
    temp = bin_img.copy()
    output = bin_img.copy()
    for j in range(bin_img.shape[0]-1,-1,-1):
        for i in range(0,bin_img.shape[1]):
            if temp[j][i] == 1:
                for s in structcross:
                    if i + s[0] >= 0 and i + s[0] < bin_img.shape[0] and i + s[1] >= 0 and i + s[1] < bin_img.shape[1]:
                        output[i+s[0]][j+s[1]] = 1
    return output

def cellcomplex(bin_img):   
    cellCC = np.zeros((bin_img.shape))
    height = bin_img.shape[0]
    width = bin_img.shape[1]   
    xindex = np.zeros((height-1,width-1))
    yindex = np.zeros((height-1,width-1))
    pointlabel = 1
    edgelabel = 1
    cell1 = []
    cell2 = []
    cell3 = []
    
    for i in range(height):
        for j in range(width):
            if bin_img[i][j] == 1:
                cell1.append([i,j])
                cellCC[i][j] = pointlabel
                edgelabel += 1
                
    for i in range(height-1):
        for j in range(width-1):
            if bin_img[i][j] == 1 and bin_img[i][j+1] == 1:
                cell2.append([cellCC[i][j],cellCC[i][j+1]])
                yindex[i][j] = edgelabel
                edgelabel += 1
            if bin_img[i][j] == 1 and bin_img[i+1][j] == 1:
                cell2.append([cellCC[i][j],cellCC[i+1][j]])
                xindex[i][j] = edgelabel
                edgelabel += 1
                
    for i in range(height-2):
        for j in range(width-2):
            if xindex[i][j] != 0 and xindex[i][j+1] != 0 and yindex[i][j] != 0 and yindex[i+1][j] != 0:
                cell3.append([xindex[i][j],xindex[i][j+1],yindex[i][j],yindex[i+1][j]])

    return cell1,cell2,cell3

def thinning(bin_img,cell1,cell2,cell3,absthreshold,relthreshold):
    removed1 = [0]*len(cell1)
    removed2 = [0]*len(cell2)
    removed3 = [0]*len(cell3)
    flag = True
    thiniter = 0
    iso = [0]*len(cell2)
    
    while flag:
        flag = False
        thiniter += 1
        cell1parents = [0]*len(cell1)
        cell2parents = [0]*len(cell2)
        
        for i in range(len(cell2)):
            if removed2[i] == 0:
                cell1parents[int(cell2[i][0]-1)] += 1
                cell1parents[int(cell2[i][1]-1)] += 1
                
        for i in range(len(cell3)):
            if removed3[i] == 0:
                cell2parents[int(cell3[i][0]-1)] += 1
                cell2parents[int(cell3[i][1]-1)] += 1
                cell2parents[int(cell3[i][2]-1)] += 1
                cell2parents[int(cell3[i][3]-1)] += 1
                
        for i in range(len(cell2)):
            if removed2[i] == 0:
                if not (cell2parents[i]==0 and (thiniter-iso[i]) > absthreshold and (1-iso[i]//thiniter) > relthreshold):
                    if cell1parents[int(cell2[i][0]-1)] == 1 and removed1[int(cell2[i][0]-1)] == 0:
                        removed2[i] = 1
                        removed1[int(cell2[i][0]-1)] = 1
                        flag = True
                    elif cell1parents[int(cell2[i][1]-1)] == 1 and removed1[int(cell2[i][1]-1)] == 0:
                        removed2[i] = 1
                        removed1[int(cell2[i][1]-1)] = 1
                        flag = True
                        
        for i in range(len(cell3)):
            if removed3[i] == 0:
                cell3elem = cell3[i]
                for j in range(4):
                    if cell2parents[int(cell3elem[j]-1)] == 1 and removed2[int(cell3elem[j]-1)] == 0:
                        removed3[i] = 1
                        removed2[int(cell3elem[j]-1)] = 1
                        flag = True
                        break
                    
        for i in range(len(cell2)):
            if removed2[i] == 0 and cell2parents[i] == 0 and iso[i] == 0:
                iso[i] = thiniter
                
    output = []
    
    for i in range(len(cell1)):
        if removed1[i] == 0:
            output.append(cell1[i])
            
    return output,removed1
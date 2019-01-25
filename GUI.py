#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Nigel
"""
import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfilename
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import tkinter.simpledialog
from numpy.linalg import inv
from skimage.io import imread, imsave
from os.path import normpath as fn
from collections import deque


class DisplayImage:

    def __init__(self, master):        
        self.master = master
        master.title("Fruitfly Sperm Cell Measurement GUI")
#        creating frames: image, text, buttons
        self.image_frame = tk.Frame(master, borderwidth=0, highlightthickness=0, height=20, width=30, bg='white')
        #        fixing image frame size
#        self.image_frame.pack(expand=True, fill='both')
#        self.image_frame.pack_propagate(0)
#        self.image_frame.grid(sticky='nswe')
#        self.image_frame.rowconfigure(0, weight=1)
#        self.image_frame.columnconfigure(0, weight=1)
#        self.image_frame.grid_propagate(0)
#        self.image_frame.pack()
        self.image_frame.grid(row=2,column=1)
        #        creating label on the frame
        self.image_label = tk.Label(self.image_frame, highlightthickness=0, borderwidth=0)
        self.image_label.pack()
        self.image_frame2 = tk.Frame(master, borderwidth=0, highlightthickness=0, height=20, width=30, bg='white')
        self.image_frame2.grid(row=2,column=2)
        #        creating label on the frame
        self.image_label2 = tk.Label(self.image_frame2, highlightthickness=0, borderwidth=0)
        self.image_label2.pack()
        self.text_frame = tk.Frame(master, borderwidth=0, highlightthickness=0,height=5, width=25, bg='white')
        self.text_frame.grid(row=0,column=0,columnspan=3)
        self.text_label = tk.Label(self.text_frame,highlightthickness=0, borderwidth=0)
        self.text_label.pack()
        self.button_frame = tk.Frame(master, borderwidth=0, highlightthickness=0, height=10, width=15, bg='yellow')
        self.button_frame.grid(row=2,column=0, sticky = N)
        self.button_label = tk.Label(self.button_frame,text="Menu",  highlightthickness=0, borderwidth=0,bg='yellow')
        self.button_label.pack()
#        status message frame
        self.status_frame = tk.Frame(master, borderwidth=0, highlightthickness=0,height=5, width=25, bg='yellow')
        self.status_frame.grid(row=1,column=0,columnspan=3, sticky = E)
        self.status_label = tk.Label(self.status_frame,highlightthickness=0, borderwidth=0)
        self.status_label.pack()
        self.home = tk.Button(self.button_frame, command=self.stopclick, text="Start Over", width=20, default=ACTIVE, borderwidth=0)
        self.home.pack()
        self.dim = tk.Button(self.button_frame, command=self.calc_dim, text="Image Dimensions", width=20, default=ACTIVE, borderwidth=0)
        self.dim.pack()
        self.newsize = tk.Button(self.button_frame, command=self.resize, text="Resize Image", width=20, default=ACTIVE, borderwidth=0)
        self.newsize.pack()
        self.zoom = tk.Button(self.button_frame, command=self.zoom_image, text="Zoom", width=20, default=ACTIVE, borderwidth=0)
        self.zoom.pack()
        self.cell = tk.Button(self.button_frame, command=self.select_cell, text="View Pixel Value", width=20, default=ACTIVE, borderwidth=0)
        self.cell.pack()
        self.brightness = tk.Button(self.button_frame, command=self.adj_brightness, text="Adjust Brightness", width=20, default=ACTIVE, borderwidth=0)
        self.brightness.pack()
        self.contrast = tk.Button(self.button_frame, command=self.adj_contrast, text="Adjust Contrast", width=20, default=ACTIVE, borderwidth=0)
        self.contrast.pack()
        self.adcontrast = tk.Button(self.button_frame, command=self.adaptivecontrast, text="Apply Adaptive Contrast", width=20, default=ACTIVE, borderwidth=0)
        self.adcontrast.pack()
        
        self.points = tk.Button(self.button_frame, command=self.select_points, text="Crop", width=10, default=ACTIVE, borderwidth=0)
        self.points.pack()
        self.erase = tk.Button(self.button_frame, command=self.select_erasePts, text="Eraser", width=10, default=ACTIVE, borderwidth=0)
        self.erase.pack()
        self.draw = tk.Button(self.button_frame, command=self.select_drawPts, text="Draw", width=10, default=ACTIVE, borderwidth=0)
        self.draw.pack()
        self.thresh = tk.Button(self.button_frame, command=self.thresholding, text="1.Threshold Image", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.thresh.pack()
        self.numComp = tk.Button(self.button_frame, command=self.numcomponents, text="2.Label Components", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.numComp.pack()
        self.largestComp = tk.Button(self.button_frame, command=self.getlargest, text="3.Get Largest Component", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.largestComp.pack()
        self.cc = tk.Button(self.button_frame, command=self.cellcomplex, text="4.Build Cell Complex", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.cc.pack()
        self.thin = tk.Button(self.button_frame, command=self.thinning, text="5.Thin Image", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.thin.pack()
        self.compare = tk.Button(self.button_frame, command=self.compareOut, text="6.Compare Output", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.compare.pack()
        self.cellLength = tk.Button(self.button_frame, command=self.measurelength, text="7.Measure Cell Length", width=20, default=ACTIVE, borderwidth=0, anchor="w")
        self.cellLength.pack()
        
        global cropstartoverflag
        cropstartoverflag=False
#        self.calc_dimensions.insert(END,"Dimensions")

    def display_image(self, event=None):
        self.imgt = ImageTk.PhotoImage(image=self.img)
#        , size=(512,512)
#    
        self.image_label.configure(image=self.imgt)

    def read_image(self, event=None):
#        select image file to open
        File = askopenfilename(parent=self.master, initialdir="./",title='Select an image')
        self.img = Image.open(File)
        self.w,self.h = self.img.size
        self.original = self.img
#        resizing image for beter display
        if(self.w>512 and self.h>512):
            self.img = self.img.resize((512,512),Image.ANTIALIAS)
        self.img.save("resized.png")
        global cropPts,cropbuttonflag,cropstartoverflag, cellPts, ocontbuttonflag, obrightbuttonflag, ocontflag, zoomflag,threshflag, drawflag, eraseflag
        cropPts=[]
        cellPts=[]
        cropbuttonflag=False
        cropstartoverflag=False
        self.brightness = 50
        ocontbuttonflag = False
        ocontflag=False
        obrightbuttonflag= False
        zoomflag = False
        threshflag = False
        global labelflag
        labelflag = False
        drawflag = False
        eraseflag = False
        self.contrast = 'default'
        self.master.after(10, self.display_image)
        self.status_label.configure(text="Status: Complete")
        
    def calc_dim(self, event=None):
        imageShape = np.asarray(self.img).shape
        self.dim = ' '+str(imageShape[0])+\
        ' x '+str(imageShape[1])+' x '+str(imageShape[2])+' (scaled 0.25x)'+'\nOriginal: '+\
        str(self.h)+' x '+str(self.w)+' x 3'
        self.status_label.configure(text="Status: Running")
#        calls function after 10 ms
        self.master.after(10,self.display_dimensions)
    
    def display_dimensions(self, event=None):
        self.text_label.configure(text=self.dim)
        self.status_label.configure(text="Status: Complete")
        
    def getcoordinates(self, event):
        cropPts.append((event.x,event.y))
        print(cropPts)
        im = self.img
        draw = ImageDraw.Draw(im) 
        draw.line(cropPts, fill='yellow')
        self.imgt = ImageTk.PhotoImage(image=im)
        self.image_label.configure(image=self.imgt)
        global cropbuttonflag
        if cropbuttonflag==True:
            self.crop = tk.Button(self.text_frame, command=self.crop_poly, text="Crop Image", width=10, default=ACTIVE, borderwidth=0)
            self.crop.pack()
            cropbuttonflag=False
    
    def select_points(self, event=None):
        tk.messagebox.showinfo("Instructions", "Click Points to Select Region on the Image")
        global buttonid,cropbuttonflag,cropstartoverflag
        buttonid = self.image_label.bind("<Button-1>",self.getcoordinates)
        cropstartoverflag=True
        if cropbuttonflag==False:
            cropbuttonflag=True
    
    def stopclick(self, event=None): #this is essentially Start Over!
        tk.messagebox.showinfo("Message", "Starting Over...")
        global ocontbuttonflag, obrightbuttonflag, drawid,drawflag, eraseid, eraseflag
        if cropstartoverflag==True:
            self.image_label.unbind("<Button-1>",buttonid)
            cropstartoverflag==False
            self.crop.destroy()
        if ocontbuttonflag==True:
            self.contrast_o.destroy()
            ocontbuttonflag=False
        if obrightbuttonflag==True:
            self.brightness_o.destroy()
            obrightbuttonflag=False
        if drawflag == True:
            self.image_label.unbind("<B1-Motion>", drawid)
            drawflag = False
        if eraseflag == True:
            self.image_label.unbind("<B1-Motion>", eraseid)
            eraseflag = False
        global cropPts, cropbuttonflag, cellPts, zoomflag, threshflag
        cellPts=[]
        cropbuttonflag=False
        self.text_label.configure(text='')
        self.brightness = 50
        self.contrast = 'default'
        ocontbuttonflag=False
        obrightbuttonflag = False
        zoomflag= False
        threshflag = False
        global labelflag
        labelflag = False
        self.status_label.configure(text="Status: Complete")
        self.master.after(10,self.read_image)
        self.image_frame2.destroy()
        

    def crop_poly(self, event=None):
        tk.messagebox.showinfo("Instructions", "Crop from Original Input Size for Accurate Scaling & Measurement")
        self.status_label.configure(text="Status: Running")
        convertImg = np.asarray(self.img)
        polygon = cropPts
        maskIm = Image.new('L', (convertImg.shape[1], convertImg.shape[0]), 0)
        ImageDraw.Draw(maskIm).polygon(polygon, outline='black', fill=1)
        mask = np.array(maskIm)
        unitmask =mask
        unitmask[unitmask>0] = 1
        # construct new image (uint8: 0-255)
        newIm = np.empty(convertImg.shape,dtype='uint8')
        # colors (RGB)
        newIm[:,:,:3] = convertImg[:,:,:3]
        newIm[:,:,0] *= unitmask
        newIm[:,:,1] *= unitmask
        newIm[:,:,2] *= unitmask
        
        bounding_rect = cv2.boundingRect(np.asarray(polygon))
        x,y,w,h = bounding_rect
        newIm = newIm[y:y+h, x:x+w].copy()
        
        # back to Image from numpy
        cropImg = Image.fromarray(newIm, "RGB")
        self.cropped = cropImg
        cropImg.save(fn('outputs/cropout.png'))
        tk.messagebox.showinfo("Instructions", "Cropped Img Saved!")
        global cropstartoverflag
        cropstartoverflag=False
        self.img = cropImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        self.image_label.unbind("<Button-1>",buttonid)
        self.crop.destroy()
#        self.master.after(10,self.stopclick)
        
        
    def zoom_image(self, event=None):
#        take new height and width inputs
        h,w= self.original.size
        newH = tk.simpledialog.askinteger("Input", "Input New Height",parent=self.master,minvalue=1,maxvalue=2048)
        newW = tk.simpledialog.askinteger("Input", "Input New Width",parent=self.master,minvalue=1,maxvalue=2048)
#        rescale image
        tk.messagebox.showinfo("Instructions", "Zooming Image... You shouldn't Crop here.")
        self.img = self.img.resize((newW,newH),Image.ANTIALIAS)
        self.text_label.configure(text='')
#        display the zoomed image
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
#        self.button_frame.place(x=newW,y=200)
        global zoomflag
        if zoomflag ==False:
            self.zoomback = tk.Button(self.text_frame, command=self.display_original, text="Original Image", width=25, default=ACTIVE, borderwidth=0)
            self.zoomback.pack()
            zoomflag=True
    
    def display_original(self, event=None):
        self.imgt = ImageTk.PhotoImage(image=self.original.resize((512,512),Image.ANTIALIAS))
        self.image_label.configure(image=self.imgt)
        self.zoomback.destroy()
        global zoomflag
        zoomflag = False
        
    def select_cell(self, event=None):
        tk.messagebox.showinfo("Instructions", "Click Points on the Image")
        global clickid
        clickid = self.image_label.bind("<Button-1>",self.getIntensity)
        
    def getIntensity(self, event):
        grayI = np.sum(np.asarray(self.img)[event.y,event.x])/3
#        print(event.x,event.y)
        cellPts.append(grayI)
        self.text_label.configure(text=cellPts)
        self.status_label.configure(text="Status: Complete")
    
    def adj_brightness(self, event=None):
        self.status_label.configure(text="Status: Running")
#        get new brightness input
        newBrightness = tk.simpledialog.askinteger("Input", "Current Brightness: "+str(self.brightness)+"%\nEnter a New Brightness(0-100%): ",parent=self.master,minvalue=0,maxvalue=100)
        newImg = np.asarray(self.img, dtype=np.uint16)
        self.prevbrightnessImg = self.img
#        scaling up the brightness
        newImg = newImg + (newBrightness-int(self.brightness))*2
#        update current brightness
        self.brightness = newBrightness
#        adjust the upper brightness limit
        newImg[newImg>255] = 255
        newImg = np.asarray(newImg, dtype=np.uint8)
        newImg = Image.fromarray(newImg, "RGB")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        global obrightbuttonflag
        if obrightbuttonflag ==False:
            self.brightness_o = tk.Button(self.text_frame, command=self.original_brightness, text="Original Brightness", width=25, default=ACTIVE, borderwidth=0)
            self.brightness_o.pack()
            obrightbuttonflag=True
    
    def original_brightness(self, event=None):
        tk.messagebox.showinfo("Instructions", "Adjusting Back to Original Brightness: 50%")
        self.img = self.prevbrightnessImg
        if(self.img.size[0]>512 and self.img.size[1]>512):
            self.img = self.img.resize((512,512),Image.ANTIALIAS)
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        self.brightness_o.destroy()
        global obrightbuttonflag
        obrightbuttonflag=False
    
    def adj_contrast(self, event=None):
        self.status_label.configure(text="Status: Running")
#        get new contrast input
        newContrast = tk.simpledialog.askinteger("Input", "Current Contrast: "+self.contrast+"\nEnter a New Contrast Threshold(0-255): ",parent=self.master,minvalue=0,maxvalue=255)
        newImg = np.asarray(self.img, dtype=np.int16)
        self.prevcontrastImg = self.img
#        scaling up the contrast
        newImg[newImg>int(newContrast)] += 30
        newImg[newImg<int(newContrast)] -= 100
        self.Contrast = newContrast
#        adjust the upper brightness limit
        newImg[newImg>255] = 255
        newImg[newImg<0] = 0
        newImg = np.asarray(newImg, dtype=np.uint8)
        newImg = Image.fromarray(newImg, "RGB")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        global ocontbuttonflag
        if ocontbuttonflag ==False:
            self.contrast_o = tk.Button(self.text_frame, command=self.original_contrast, text="Original Contrast", width=25, default=ACTIVE, borderwidth=0)
            self.contrast_o.pack()
            ocontbuttonflag = True
        
    def original_contrast(self, event=None):
        tk.messagebox.showinfo("Instructions", "Adjusting Back to Original Contrast: default")
        self.img = self.prevcontrastImg
        if(self.img.size[0]>512 and self.img.size[1]>512):
            self.img = self.img.resize((512,512),Image.ANTIALIAS)
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        self.contrast_o.destroy()
        global ocontbuttonflag
        ocontbuttonflag=False

    def adaptivecontrast(self,event=None):
        tk.messagebox.showinfo("Instructions", "Adaptive Contrasting...")
        self.status_label.configure(text="Status: Running")
        self.prevcontrastImg = self.img
        k = 5 #3,4
        contrastval = 60 #40,50
        tempimg = np.asarray(self.img).copy()
        tempimg1 = tempimg[:,:,0]
        img = np.asarray(self.img).astype(int).copy()
        print("Adaptive Contrast Start")
        for i in range(tempimg.shape[0]):
            for j in range(tempimg.shape[1]):
                isum = 0
                count = 0
                for ki in range(-k,k+1):
                    for kj in range(-k,k+1):
                        if ki+i < tempimg.shape[0] and ki+i >= 0 and kj+j < tempimg.shape[1] and kj+j >= 0:
                           if np.all(img[ki+i][kj+j]) == False:
                               pass
                           else:
                               isum += tempimg1[ki+i][kj+j]
                               count += 1
                if tempimg1[i][j] * count > isum:
                        if tempimg1[i][j] + contrastval > 255:
                            tempimg[i][j] = [255,255,255]
                        else:
                            tempimg[i][j] = img[i][j] + contrastval
        imsave(fn('outputs/adContrastOut.png'),tempimg)
        
        newImg = Image.fromarray(tempimg, 'RGB')
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Complete")
        print("Adaptive Contrast Done")
        global ocontbuttonflag
        if ocontbuttonflag ==False:
            self.contrast_o = tk.Button(self.text_frame, command=self.original_contrast, text="Original Contrast", width=25, default=ACTIVE, borderwidth=0)
            self.contrast_o.pack()
            ocontbuttonflag = True

        
    def thresholding(self,event=None):
        tk.messagebox.showinfo("Instructions", "Thresholding...")
        self.status_label.configure(text="Status: Thresholding...")
        k=5
        npImg = np.asarray(self.img)
        grayImg = cv2.cvtColor(npImg, cv2.COLOR_BGR2GRAY)
        self.grayImg = grayImg
        bin_img = np.zeros((grayImg.shape), dtype=int)
        print("Thresholding Start")
        for i in range(grayImg.shape[0]):
            for j in range(grayImg.shape[1]):
                ksum = 0
                count = 0
                if grayImg[i][j] != 0:
                    for ki in range(-k,k+1):
                        for kj in range(-k,k+1):
                            if ki+i < grayImg.shape[0] and ki+i >= 0 and kj+j < grayImg.shape[1] and kj+j >= 0:
                                if grayImg[ki+i][kj+j] != 0:
                                    ksum += grayImg[ki+i][kj+j]
                                    count += 1
                    if grayImg[i][j]*count > ksum:
                        bin_img[i][j] = 1
        self.bin_img = bin_img
#        print(np.unique(self.bin_img))
        tempImg =bin_img.copy()
        tempImg[tempImg==1] = 255
        newIm=tempImg.astype('uint8')
        imsave(fn('outputs/threshout.png'),tempImg)
        
        newImRGB = np.stack((newIm,)*3, axis=-1)
        newImg = Image.fromarray(newImRGB, 'RGB')
        print("Thresholding Done")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Thresholding Complete")
        global threshflag
        threshflag = True
        
        
    def numcomponents(self,event=None):
        img = self.bin_img
        labels = np.zeros((img.shape),dtype=int)
        vflag = np.zeros((img.shape),dtype=int)
        countlabel=0
        print("NumComponents Start")
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if vflag[i][j] == 1:
                    continue
                countlabel +=1
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
                    labels[x][y] = countlabel
                    queue_points.append([x-1,y])
                    queue_points.append([x+1,y])
                    queue_points.append([x,y-1])
                    queue_points.append([x,y+1])
                    queue_points.append([x-1,y-1])
                    queue_points.append([x-1,y+1])
                    queue_points.append([x+1,y-1])
                    queue_points.append([x+1,y+1])
                    
        labels=labels.astype('uint8')
        imsave(fn('outputs/numcomponents.png'),labels)
        newImRGB = np.stack((labels,)*3, axis=-1)

        newImg = Image.fromarray(newImRGB,'RGB')
        print("NumComponents Done")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: Labeling Components Complete")
        self.nComp = countlabel
        self.labels=labels
        global labelflag
        labelflag == True
        
    def getlargest(self):
        print("Largest Components Start")
        k = tk.simpledialog.askinteger("Input", "How many Largest Components? ",parent=self.master,minvalue=0,maxvalue=20)

        nComp = self.nComp
        bin_img = self.bin_img.astype(int)
        tempimg = np.zeros((bin_img.shape),dtype=int)
        print(np.unique(bin_img))
        labels = self.labels
        counts = np.zeros((nComp),dtype=int)
        for i in range(bin_img.shape[0]):
            for j in range(bin_img.shape[1]):
                counts[labels[i][j]] += 1
        counts = counts.tolist()
        whilecount=0
        while whilecount<k:
            maxi = 1
            maxs = counts[maxi]
            for i in range(2,nComp):
                if counts[i] > maxs:
                    maxi = i
                    maxs = counts[i]
            for x in range(bin_img.shape[0]):
                for y in range(bin_img.shape[1]):             
                    if labels[x][y] == maxi:
                        tempimg[x][y] = 1
            counts[maxi] = 0
            whilecount+=1
                    
        self.bin_img = tempimg
        
        tempImg = tempimg.copy()
        tempImg[tempImg==1] = 255
        newIm=tempImg.astype('uint8')
        imsave(fn('outputs/largestcomponents.png'),tempImg)
        newImRGB = np.stack((newIm,)*3, axis=-1)
        newImg = Image.fromarray(newImRGB, 'RGB')
        print("Largest Components Done")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.status_label.configure(text="Status: K Largest Components Complete")
#        print(np.unique(bin_img))
    def cellcomplex(self):
        print("Build Cell Complex Start")
        bin_img = self.bin_img
        height = bin_img.shape[0]
        width = bin_img.shape[1] 
        cellCC = np.zeros((bin_img.shape))
        xindex = np.zeros((bin_img.shape[0]-1,bin_img.shape[1]-1))
        yindex = np.zeros((bin_img.shape[0]-1,bin_img.shape[1]-1))
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
                    pointlabel += 1
                    
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
                if xindex[i][j] > 0 and xindex[i][j+1] > 0 and yindex[i][j] > 0 and yindex[i+1][j] > 0:
                    cell3.append([xindex[i][j],xindex[i][j+1],yindex[i][j],yindex[i+1][j]])
                    
        self.cell1 = cell1
        self.cell2 = cell2
        self.cell3 = cell3
        print("Build Cell Complex Done")
        self.status_label.configure(text="Status: Build Cell Complex Complete")
    
    def thinning(self):
        self.status_label.configure(text="Status: Thinning Start")
        print("Thinning Start")
        bin_img = self.bin_img
        cell1 = self.cell1
        cell2 = self.cell2
        cell3 = self.cell3
        absthreshold = 5
        relthreshold = 0.5
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
                    iso[i] = thiniter+1
                    
        output = []
        
        for i in range(len(cell1)):
            if removed1[i] == 0:
                output.append(cell1[i])
        outImg = np.full((bin_img.shape),0)
        
        for l in output:
            outImg[l[0]][l[1]] = 255
        
        print("Number of Thinned Output Points: ",len(output))
        outImg=outImg.astype('uint8')
        imsave(fn('outputs/thinOut.png'),outImg)
        newImRGB = np.stack((outImg,)*3, axis=-1)
        newImg = Image.fromarray(newImRGB, 'RGB')
        print("Thinning Done")
        self.img = newImg
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.final = len(output)
        self.status_label.configure(text="Status: Thinning Complete")
        
    def measurelength(self):
        len = self.final
        self.text_label.configure(text=str(len*4/3.06)+' micrometers')
        self.status_label.configure(text="Status: Measure Cell Length Complete")
        
    def select_erasePts(self):
#        self.status_label.configure(text="Status: Running")
        self.eL = tk.simpledialog.askinteger("Input", "Select a size of your eraser?(0-100, odd integer)",parent=self.master,minvalue=0,maxvalue=100)
#        self.eW = tk.simpledialog.askinteger("Input", "What is the width of your eraser?(0-100) ",parent=self.master,minvalue=0,maxvalue=100)
        global eraseid, drawflag, drawid, eraseflag
        if drawflag ==True:
            self.image_label.unbind("<B1-Motion>",drawid)
            drawflag = False
        if np.mod(self.eL,2) != 0:
            eraseid = self.image_label.bind("<B1-Motion>",self.erasePts)
            eraseflag = True
        else:
            tk.messagebox.showinfo("Instructions", "Please Read the Instruction Carefully.")

        
    def erasePts(self, event):
        global ex, ey
        ex = int(event.x)
        ey = int(event.y)
#        eraser = np.array((self.eL,self.eL))
#        eraser = eraser[:,:,np.newaxis]
        eraserRGB = np.full((self.eL,self.eL,3),0)
#        eraser[:,:,0] = [0,0,0]
        spanL = int((self.eL-1)/2)
#        print(type(ex),type(ey),type(spanL))
        im = np.asarray(self.img).copy()
        h,w,c = im.shape
        imPad = np.pad(im, ((spanL,spanL),(spanL,spanL),(0,0)),'constant',constant_values=0)
        imPad[ey-spanL:ey+spanL+1, ex-spanL:ex+spanL+1] = eraserRGB
        print((ex,ey))
        global threshflag,labelflag
        if threshflag == True:
            eraserBin = np.zeros((self.eL,self.eL))
            binImg = self.bin_img.copy()
            bh,bw = binImg.shape
            binPad = np.pad(binImg, (spanL,spanL),'constant',constant_values=0)
            binPad[ey-spanL:ey+spanL+1, ex-spanL:ex+spanL+1] = eraserBin
            self.bin_img = binPad[spanL:spanL+bh, spanL:spanL+bw]
        if labelflag == True:
            eraserBin = np.zeros((self.eL,self.eL))
            limg = self.labels.copy()
            lh,lw = limg.shape
            lPad = np.pad(limg, (spanL,spanL),'constant',constant_values=0)
            lPad[ey-spanL:ey+spanL+1, ex-spanL:ex+spanL+1] = eraserBin
            self.labels = lPad[spanL:spanL+lh, spanL:spanL+lw]
#        draw = ImageDraw.Draw(im) 
#        draw.line(cropPts, fill='yellow')
        im = imPad[spanL:spanL+h, spanL:spanL+w,:]
        erasedIm = Image.fromarray(im, 'RGB')
        self.img = erasedIm
        self.imgt = ImageTk.PhotoImage(image=erasedIm)
        self.image_label.configure(image=self.imgt)
        
    def select_drawPts(self):
        self.bL = tk.simpledialog.askinteger("Input", "Select a size of your brush?(0-20, odd integer)",parent=self.master,minvalue=0,maxvalue=20)
#        self.eW = tk.simpledialog.askinteger("Input", "What is the width of your eraser?(0-100) ",parent=self.master,minvalue=0,maxvalue=100)
        global drawid, drawflag,eraseid, eraseflag
        if eraseflag ==True:
            self.image_label.unbind("<B1-Motion>",eraseid)
            eraseflag = False
        if np.mod(self.bL,2) != 0:
            drawid = self.image_label.bind("<B1-Motion>",self.drawPts)
            drawflag = True
        else:
            tk.messagebox.showinfo("Instructions", "Please Read the Instruction Carefully.")
    
    def drawPts(self, event):
        global bx, by
        bx = int(event.x)
        by = int(event.y)

        brushRGB = np.full((self.bL,self.bL,3),255)
        spanL = int((self.bL-1)/2)
#        print(type(ex),type(ey),type(spanL))
        im = np.asarray(self.img).copy()
        h,w,c = im.shape
        imPad = np.pad(im, ((spanL,spanL),(spanL,spanL),(0,0)),'constant',constant_values=0)
        imPad[by-spanL:by+spanL+1, bx-spanL:bx+spanL+1] = brushRGB
        print((bx,by))
        global threshflag,labelflag
        if threshflag == True:
            brushBin = np.full((self.bL,self.bL),255)
            binImg = self.bin_img.copy()
            bh,bw = binImg.shape
            binPad = np.pad(binImg, (spanL,spanL),'constant',constant_values=0)
            binPad[by-spanL:by+spanL+1, bx-spanL:bx+spanL+1] = brushBin
            self.bin_img = binPad[spanL:spanL+bh, spanL:spanL+bw]
#            print("working")
        if labelflag == True:
            brushBin = np.full((self.bL,self.bL),255)
            limg = self.labels.copy()
            lh,lw = limg.shape
            lPad = np.pad(limg, (spanL,spanL),'constant',constant_values=0)
            lPad[by-spanL:by+spanL+1, bx-spanL:bx+spanL+1] = brushBin
            self.labels = lPad[spanL:spanL+lh, spanL:spanL+lw]
#        draw = ImageDraw.Draw(im) 
#        draw.line(cropPts, fill='yellow')
        im = imPad[spanL:spanL+h, spanL:spanL+w,:]
        drawnIm = Image.fromarray(im, 'RGB')
        self.img = drawnIm
        self.imgt = ImageTk.PhotoImage(image=drawnIm)
        self.image_label.configure(image=self.imgt)
        
    def compareOut(self):
        self.image_frame2 = tk.Frame(self.master, borderwidth=0, highlightthickness=0, height=20, width=30, bg='white')
        self.image_frame2.grid(row=2,column=2)
        #        creating label on the frame
        self.image_label2 = tk.Label(self.image_frame2, highlightthickness=0, borderwidth=0)
        self.image_label2.pack()
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.imgc = ImageTk.PhotoImage(image=self.cropped)
        self.image_label2.configure(image=self.imgc)
    
    def resize(self):
        newh = tk.simpledialog.askinteger("Input", "Type in the new height: ",parent=self.master,minvalue=0,maxvalue=self.img.size[0])
        neww = tk.simpledialog.askinteger("Input", "Type in the new width: ",parent=self.master,minvalue=0,maxvalue=self.img.size[1])
        self.img = self.img.resize((neww,newh),Image.ANTIALIAS)
        self.imgt = ImageTk.PhotoImage(image=self.img)
        self.image_label.configure(image=self.imgt)
        self.img.save("resized.png")
def main():
    root = tk.Tk()
    GUI = DisplayImage(root)
    GUI.read_image()
    root.mainloop()

if __name__ == '__main__':
    main()
#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
from glob import glob
import random
# import nia22
import json
import cv2
import matplotlib.pyplot as plt 
import datetime

import utils

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

#get_dir
nas_dir='/run/user/1000/gvfs/ftp:host=192.168.0.43/NIA2022/raw/'
#date = ['1018','1025','1031','1101','1102']
#devices = ["Laptop", "Monitor", "VehicleLCD", "Smartphone", "Tablet"]
img_size=(224,224)

#with open(nas_dir+"dates.txt", "r") as f:
#    ll = f.read().split("\n")
#    ll = [l for l in ll if len(l) > 2 ]

with open("set_1.txt", "r") as f:
    val_set = f.read().split("\n")
    val_set.pop()

#val_set_s = [vs for vs in val_set if vs.split("/")[13] == "Smartphone"]
#val_set_v = [vs for vs in val_set if vs.split("/")[13] == "VehicleLCD"]
#val_set_m = [vs for vs in val_set if vs.split("/")[13] == "Monitor"]
#val_set_l = [vs for vs in val_set if vs.split("/")[13] == "Laptop"]
#val_set_t = [vs for vs in val_set if vs.split("/")[13] == "Tablet"]
#val_set_s = [vs for vs in val_set if vs.split("/")[13] == "Smartphone"]
#val_set_s = [vs for vs in val_set if vs.split("/")[13] == "Smartphone"]
#val_set_s = [vs for vs in val_set if vs.split("/")[13] == "Smartphone"]
#lists_device = {'Laptop':val_set_l,
#                'VehicleLCD':val_set_v,
#                'Monitor':val_set_m,
#                'Smartphone':val_set_s,
#                'Tablet':val_set_t
#                }

#base = "/media/di2/T7/"
base = './'
#val_set = glob(base+"*.mp4")
print(len(val_set))
nframes=32
sampling_freq=8
#for device in devices[:3]:
#if True:
video_last=12132
with open("annotation/set_2.txt", "w") as f:
    out_base = base+ f"all_clips/"
    #video_last = len(glob(base + f"all_clips/v*/frame_15.png"))
    #n_lists = len(glob(base+"clip_list*.txt"))
    os.makedirs(out_base,exist_ok=True)
    for fn_video in val_set:
        #print("-------------------",device,"-------------------------------")

            if True:
            #for fn_video in lists_device[device]:
                print(fn_video)
                state = utils.get_state(fn_video)
                dir_out = out_base + f"v{video_last:06d}/"
                print(dir_out)
                if not os.path.isdir(dir_out):   #// This fails
                    os.makedirs(dir_out)        
                try:
                #if True:
                    nframes_done = utils.crop_frames(fn_video, dir_out, 
                                nframes=nframes, 
                                sampling_freq = sampling_freq, 
                                #rot=cv2.ROTATE_90_COUNTERCLOCKWISE,
                                factor=2.0)
                    video_last +=1
                    #if nframes_done < nframes:
                    #    continue            
                    f.write(f"{dir_out} {nframes_done} {state}\n")
                    print("done", video_last, "state", state)
                    f.flush()
                except:
                #else:
                    continue
        #f.close()

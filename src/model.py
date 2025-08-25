import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

data_dir = 'data/GTSRB/Train'
classes = os.listdir(data_dir)
print("Number of classes:", len(classes))



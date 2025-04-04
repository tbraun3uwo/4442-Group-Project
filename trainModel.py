from fastai.vision.all import *
import pathlib
import os
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

path = "data/dataset"
dls = ImageDataLoaders.from_folder(path, valid_pct=0.2, seed=42, item_tfms=Resize(25))
dls.show_batch()

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(50)

categories = ("0","1","2","3","4","5","6","7","8","9","add","dec","div","eq","mul","sub")
def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

# Set the correct path for saving
learn.path = pathlib.Path('.')
learn.export('model/model.pkl')

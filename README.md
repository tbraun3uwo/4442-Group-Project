# AI-Powered Handwriting Recognition for Automated Student Work Evaluation
# Project members: Travis, Nasri, Tim, Reesa
# MathStang - an app to solve math questions
=======
## Project Members
- Timothy Nguyen
- Travis Braun
- Reesa Dayani
- Nasri Hussein

## Course Information
- Course: CS4442
- Final Project
- Date: Mar 12, 2025
- Course Instructors:
  - Dr. Boyu Wang, Dept. of Computer Science
  - Dr. Yalda Mohsenzadeh, Dept. of Computer Science

## Project Abstract

### Introduction
In a growing society, as student populations increase, teachers' workload increases, making it difficult to provide timely feedback on student assignments. As an attempt to reduce teacher workload, we propose an AI-powered application that automates the recognition, evaluation, and feedback process for student work. For this project, our app will focus on fundamental algebraic problems such as addition, subtraction, multiplication and division. By taking advantage of handwriting recognition (HWR), the app transcribes student handwritten responses to machine-readable text, performs cross-checking and evaluates student's work.

### AI Methods
Our project utilizes computer vision and Convolutional Neural Networks (CNNs) for handwriting recognition (HWR) to accurately detect and transcribe handwritten expressions. CNNs extract key features from handwriting, such as edges, curves, and stroke patterns, enabling precise character recognition, even across varied handwriting styles. The process involves image preprocessing, feature extraction, and classification, where the CNN scans, learns, and categorizes digits, operators, and variables.

### Datasets
1. Handwritten Math Symbols Dataset by Sagyam Thapa
   - Contains:
     - Digits 0-9
     - Math operators (addition, subtraction, multiplication, division)
     - Equations
     - Decimals
   - Source: [Handwritten Math Symbols by Sagyam Thapa](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols)

2. MMNIST dataset
    - Contains 70.0k handwriten digits from 0-9
    - Source: [MNIST](https://www.kaggle.com/datasets/playlist/mnistzip?fbclid=IwY2xjawJbpopleHRuA2FlbQIxMAABHYJBd2TRkWfutTysvVCP96Wj3mTmC2ki5l33pJbZkuA2SXXfVu1EWLGNmg_aem_9n4Bjp9gSy-viGtYoqVgEw)

To effectively train the model, we merge the two datasets together to create a [larger dataset here](https://uwoca-my.sharepoint.com/:u:/g/personal/knguy52_uwo_ca/ES9eNau2jsJFjAlgWAJWtXgBLIL-tsgT4GWAtaQN2Rw3HQ?e=OmdMl8)

### Running our app
- Requirement: Python 3.11
- Installing dependencies: `pip install -r requirements.txt`

- trainModel.py: fine tune available pre-trained resnet18 using Handwritten Math Symbols dataset to recognize each of the math symbols

- detectorModel.py: run the trained model and return the characterPrediction

- To fine-tune the model, please download the [dataset here](https://uwoca-my.sharepoint.com/:u:/g/personal/knguy52_uwo_ca/ES9eNau2jsJFjAlgWAJWtXgBLIL-tsgT4GWAtaQN2Rw3HQ?e=OmdMl8), untar by using command `tar -xzf mathDataset.tar.gz` and put it in the "/data" folder

- To use our already fine-tuned model, please download the [model here](https://uwoca-my.sharepoint.com/:u:/g/personal/knguy52_uwo_ca/ESDoGlpcWl9MsEfUygB8kIUBDKCVevEMgbGGr4ltZ8FT5g?e=Ddijkx) and put it in the "/model" folder (if you use the fine-tuned model, no need to download the dataset or run trainModel.py)

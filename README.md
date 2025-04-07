# AI-Powered Handwriting Recognition for Automated Student Work Evaluation
# Project members: Travis, Nasri, Tim
# MathStang - an AI-powered grading assistant for handwritten math work
=======
## Project Members
- Timothy Nguyen
- Travis Braun
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
In a growing society, as student populations increase, teachers' workload increases, making it difficult to provide timely feedback on student assignments. As an attempt to reduce teacher workload, we propose an AI-powered application that automates the recognition, evaluation, and feedback process for student work. Our app focuses on fundamental algebraic problems such as addition, subtraction, multiplication and division. By taking advantage of handwriting recognition (HWR), the app:
- Transcribes student handwritten responses to machine-readable text
- Evaluates the correctness of each step in the solution
- Identifies common errors and misconceptions
- Provides detailed feedback on where students went wrong
- Suggests areas for improvement

### AI Methods
Our project utilizes computer vision and Convolutional Neural Networks (CNNs) for handwriting recognition (HWR) to detect and transcribe handwritten expressions accurately. CNNs extract key features from handwriting, such as edges, curves, and stroke patterns, enabling precise character recognition across varied handwriting styles. The process involves:
1. Image preprocessing and feature extraction
2. Character and symbol recognition
3. Step-by-step solution analysis
4. Error detection and classification
5. Feedback generation based on identified errors

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

- trainModel.py: fine-tune the available pre-trained ResNet18 using Handwritten Math Symbols dataset to recognize each of the math symbols

- detectorModel.py: run the trained model to:
  - Recognize handwritten math work
  - Evaluate solution steps
  - Generate grading feedback
  - Identify common errors

- To fine-tune the model, please download the [dataset here](https://uwoca-my.sharepoint.com/:u:/g/personal/knguy52_uwo_ca/ES9eNau2jsJFjAlgWAJWtXgBLIL-tsgT4GWAtaQN2Rw3HQ?e=OmdMl8), untar by using command `tar -xzf mathDataset.tar.gz` and put it in the "/data" folder

- To use our already fine-tuned model, please download the [model here](https://uwoca-my.sharepoint.com/:u:/g/personal/knguy52_uwo_ca/ESDoGlpcWl9MsEfUygB8kIUBDKCVevEMgbGGr4ltZ8FT5g?e=Ddijkx) and put it in the "/model" folder (if you use the fine-tuned model, no need to download the dataset or run trainModel.py)

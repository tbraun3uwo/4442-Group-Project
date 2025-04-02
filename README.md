# AI-Powered Handwriting Recognition for Automated Student Work Evaluation

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

## Datasets
1. MNIST Dataset
   - Well known for training numbers HWR
   - Contains:
     - Training set: 60,000 examples
     - Test set: 10,000 examples
   - Source: [MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data)

2. Handwritten Math Symbols Dataset
   - Contains:
     - Digits 0-9
     - Math operators (addition, subtraction, multiplication, division)
     - Equations
     - Decimals
     - Variables (x, y, z)
   - Source: [Handwritten Math Symbols on Kaggle](https://www.kaggle.com/datasets/sagyamthapa/handwritten-math-symbols)
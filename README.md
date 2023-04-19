# Feeling It: Emotion Classification with Machine Learning
*Using Convolutional Neural Networks to Detect and Analyze Human Emotions from Facial Images*

*Andrew Kroening, Chloe (Ke) Liu, Wafiakmal Miftah, Jenny (Yiran) Shen*

*Spring 2023*

**If you want to skip right to the end, [click here for the video presentation](https://youtu.be/l5mb75gyGYE)**

## Documents

The project proposal can [be found here.](/40_docs/IDS705 Project Proposal.pdf)

The final project report can be found here.

## Abstract

With the onset of the COVID-19 Pandemic, new challenges were introduced to the dynamics of human interaction. As people physically separated or wore face masks, interpreting emotional states became more challenging. Prior to this, computer vision advanced significantly with the arrival of deep learning and convolutional neural network architectures and techniques. This research leverages computer vision models to interrogate the effectiveness of machines at determining emotions from faces and makes a comparison to classical psychological research. The research also experiments with obstructing portions of the face to better approximate the challenges from poor webcam resolution, obstruction, or face masking.  We find that computer models perform reasonably well in this space; however, there are complications posed by some similar emotions, such as anger and disgust. 

## Project Overview

The onset of the COVID-19 Pandemic introduced new wrinkles to identifying emotions from observing faces. As much of the world began to socially distance or adopt facemasks as a standard practice, questions about the possible impacts on human interaction and emotional inference naturally emerged. While this challenge has already existed in certain cultures where the wearing of items that obstruct a part of a person’s face might be tradition, the COVID-19 pandemic brought this question to a whole new scale.

One of the most notable areas where this question extends today is the more prevalent use of webcams and virtual meetings. Everyone has experienced a virtual meeting where a colleague’s face might be partially blocked, poorly positioned in the view pane, or obstructed by other issues. Reading the audience’s emotions can be challenging in these environments, a critical task in human communication. With this in mind, this project will seek to gain a greater understanding of where computer vision might be able to augment human interpretation when a portion of the face is obstructed and where a model’s limitations are. 

Below is a represnetation of our experimental workflow:

<img src="/20_intermediate_files/ML Project Flowchart.jpg" width="800"/>


## Data

The facial expression dataset originated from [Kaggle at this link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset). This dataset consisted of 7 facial expressions from various gender, race, and age. In the original state of the dataset, when downloaded, we found over 35,887 images. We later removed duplicates and revised some classification labels, which resulted in a slightly altered count of each emotion.

## Exploratory Data Analysis

Our exploratory data analysis began by checking these emotional categories for balance and visual sampling to ensure that the author assigned the categories correctly. The visualization of the category balance shows some skew, while the visual inspection found multiple instances that the team considered either an erroneous assignment or a potentially duplicated image. A team member conducted an image-by-image review of the dataset to correct the erroneous categorization in the base dataset. This resulted in a slight alteration to the emotional categories, as the overall size of the dataset outweighed the small but concerning number of misclassified images. In total, we identified 1,506 images that had at least one exact copy. Anecdotally, we later observed cases where a face was present in more than one image, but the aspect ratio, zoom, or other features were altered very slightly. These cases escaped our inspection because they were not identical copies of one another. The figure below presents the dataset before and after these steps were taken.

<img src="/20_intermediate_files/ML EDA.png" width="800"/>

## Model Results

Below are the results of our model evaluations. We find that both models perform reasonably well, with some struggles in specific emotions:

<img src="/20_intermediate_files/ML Project Flowchart.jpg" width="800"/>

<img src="/20_intermediate_files/ML Project Flowchart.jpg" width="800"/>


## How-to Reproduce:
1. Download the data from [this link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) or ask us for the access to `relabeled_recleaned` dataset.
2. Run the `apply_masks.ipynb` notebook to apply the masks to the images if you are using Google Colab or `apply_masks_run_on_codespace.ipynb` if you are using GitHub Codespaces.
3. Run the `baseline vgg16.ipynb` notebook to train, validate and test the VGG16 model, and run the `baseline_resnet50.ipynb` followed by `resnet50_after_hypertune_saliency.ipynb` notebook to train, validate and test the ResNet50 model. These two notebooks will run each model with each type of data, relabeled_cleaned and all the masks.
4. Run the `vgg16_after_hypertune_saliency` for saliency map of VGG16 model, and `resnet50_after_hypertune_saliency.ipynb` for saliency map of ResNet50 model.
5. To use the real-time webcam, use the `real-time-detection.py` and make sure you have all the required data in the same directory as the python file. 
6. Enjoy! :smile:

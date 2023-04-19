### This directory is the host location for the source data, which can be found at [this link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

### The data is organized into 2 folders (train and validation), and within each folder, there are 7 subfolders, one for each expression

### All source data is stored in this directory which is ignored by git. Any transformation will call this location and subsequently move data to [the cleaned data directory](../05_clean_data)

### Reproducibility:
1. Download the data from [this link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset) or ask us for the access to `relabeled_recleaned` dataset.
2. Run the `apply_masks.ipynb` notebook to apply the masks to the images if you are using Google Colab or `apply_masks_run_on_codespace.ipynb` if you are using GitHub Codespaces.
3. Run the `baseline vgg16.ipynb` notebook to train, validate and test the VGG16 model, and run the `baseline_resnet50.ipynb` followed by `resnet50_after_hypertune_saliency.ipynb` notebook to train, validate and test the ResNet50 model. These two notebooks will run each model with each type of data, relabeled_cleaned and all the masks.
4. Run the `vgg16_after_hypertune_saliency` for saliency map of VGG16 model, and `resnet50_after_hypertune_saliency.ipynb` for saliency map of ResNet50 model.
5. To use the real-time webcam, use the `real-time-detection.py` and make sure you have all the required data in the same directory as the python file. 
6. Enjoy! :smile:
# Classifying Multimodal Data using Transformers

## Motivation
The increasing prevalence of multimodal data in our society has led to the increased need for machines to make sense of such data holistically However, data scientists and machine learning engineers aspiring to work on such data face challenges fusing the knowledge from existing tutorials which often deal with each mode separately. Drawing on our experience in classifying multimodal municipal issue feedback in the Singapore government, we conduct a hands-on tutorial to help flatten the learning curve for practitioners who want to apply machine learning to multimodal data.

## Dataset
Unfortunately, we are not able to conduct the tutorial using the municipal issue feedback data due to its sensitivity. Instead, we use a subset of the WebVision data. This dataset consists of labelled images, together with descriptions of them, crawled from the web. We chose this dataset because of its similar characteristics to our municipal issue feedback data (text descriptions correlate highly with the labels but associated images provide even better context).

## Tutorial Outline
In this tutorial, we teach participants how to classify multimodal data consisting of both text and images using Transformers. It
is targeted at an audience who have some familiarity with neural networks and are comfortable with writing code.

The outline of the tutorial is as follows:
1. Sharing of Experience: Municipal issue feedback classification in the Singapore government
2. Text Classification: Train a text classification model using BERT
3. Text and Image Classification (v1): Train a dual-encoder text and image classification model using BERT and ResNet-50
4. Text and Image Classification (v2): Train a joint-encoder text and image classification model using Align before Fuse (ALBEF)
5. Question and Answer/Discussion


The tutorial will be conducted using [Google Colab]{https://colab.research.google.com/}. We will be using the file `multimodal_training.ipynb` for the session. To run the notebook on Colab: 
1. Fo to the GitHub option and search for `dsaidgovsg/multimodal-learning-hands-on-tutorial`
2. Select the `main` branch
3. Open `multimodal_training.ipynb`
4. Follow the instructions in the cells


## Running the Python Script
The content in the notebook is meant to be a step-by-step guide to show the difference between the difference model architectures, thus the code can be quite repetitve.

We have also streamlined the code into python script which you can run from the terminal to train the models or do prediction from pretrained models. Steps to run the scripts are as follows:
1. If you have not already done so, clone this repo to your working directory `git clone https://github.com/dsaidgovsg/multimodal-learning-hands-on-tutorial.git`
2. Inside your working directory, run `bash prepare_folders_and_download_files.sh` . The script will create the folder structure and download the files used during the tutorial into these folders.
3. Install the libraries required via `pip install -r requirements.txt`
4. To do prediction on the test set using the downloaded pretrained models trained for 20 iterations, run `python3 multimodal_testing.py`
5. To do your own training and prediction, run `python3 multimodal_training.py`. Edit the `args` dictionary in the `main` function if you want to change the training parameters.





# Classifying Multimodal Data using Transformers: Applications to Municipal Issue Feedback in the Singapore Government

In this tutorial, we will teach participants how to classify municipal issue feedback cases so that they get assigned to the agency most suited to handle the case. The data, proprietary to the government of Singapore, consists of feedback submitted by the public and each feedback consists of a textual description of the issue, and an accompanying image.

We will first teach the participants how to classify the feedback using the textual description alone using Transformers. This will be done using the Hugging Face Transformers library in PyTorch.

Next, we will use ResNet-50 to get features from the images submitted and add them to the text model to train a new text + image model. We then compare the accuracy of this model with the one using text alone.

Following that, we use a different model architecture, Align before Fuse (ALBEF) by SalesForce, which aligns the image and text representations before fusing them through cross-modal attention, and compare the model’s accuracy with the previous text and image model whose representations are unaligned.

The outline of the tutorial is given below:
- Setting up the environment and loading the data
- Data exploration - Examine samples from the dataset to understand the municipal issue feedback
- Quick introduction on how to use the Hugging Face Transformers library
- Text Classification - Train a sequence classification model using the Hugging Face Transformers library that can predict the correct handling agency, using the textual description only
- Text + Image Classification (v1) - Combine the textual model with ResNet-50 image features from the images submitted to build an improved model
- Text + Image Classification (v2) - Train another text + image model with ALBEF and compare its performance with the previous model
- Question and Answer/Discussion 

#!/bin/sh

# download webvision data files
WEBVISION_DATA_FOLDER="./KDD/webvision_data/"

if [ ! -d $WEBVISION_DATA_FOLDER ] 
then
    echo "Creating data folder and downloading data files" 
    mkdir -p $WEBVISION_DATA_FOLDER
    wget "https://drive.google.com/uc?id=1r4aTTbLuYgGrgpZLOgUH9sQ33DBsbOFm&export=download" -O $WEBVISION_DATA_FOLDER/data.zip
    unzip $WEBVISION_DATA_FOLDER/data.zip -d $WEBVISION_DATA_FOLDER
    echo "Completed" 
fi



# ALBEF_FOLDER="./KDD/albef/"

# if [ ! -d $ALBEF_FOLDER ] 
# then
#     echo "Creating albef folder and downloading model files" 
#     # download albef files
#     mkdir -p $ALBEF_FOLDER
#     wget https://storage.googleapis.com/sfr-pcl-data-research/ALBEF/ALBEF_4M.pth -O $ALBEF_FOLDER/ALBEF.pth
#     wget https://raw.githubusercontent.com/salesforce/ALBEF/main/configs/config_bert.json -O $ALBEF_FOLDER/config_bert.json
#     echo "Completed" 
# fi



# TRAINED_MODEL_FOLDER="./KDD/trained_models/"
# if [ ! -d $TRAINED_MODEL_FOLDER ] 
# then
#     echo "Creating folder and downloading model files" 
#     mkdir -p $TRAINED_MODEL_FOLDER/BERT
#     wget "https://drive.google.com/uc?id=1hkc3iXG5xgnvV9ksnSf85OHzv3vApVAB&export=download" -O $TRAINED_MODEL_FOLDER/BERT/state_dict.pt
#     mkdir -p $TRAINED_MODEL_FOLDER/BERT_ResNet
#     wget "https://drive.google.com/uc?id=1a6skPhOMHX9i93q_arJGDbEr-nrPYyTj&export=download" -O $TRAINED_MODEL_FOLDER/BERT_ResNet/state_dict.pt
#     mkdir -p $TRAINED_MODEL_FOLDER/ALBEF
#     wget "https://drive.google.com/uc?id=1cZHmAfWBX525c9b_keQ3kle70_devGwm&export=download" -O $TRAINED_MODEL_FOLDER/ALBEF/state_dict.pt
#     echo "Completed" 
# fi


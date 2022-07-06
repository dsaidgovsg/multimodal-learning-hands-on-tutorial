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



HOME_FOLDER="./KDD"
TRAINED_MODEL_FOLDER="./KDD/trained_models/"
if [ ! -d $TRAINED_MODEL_FOLDER ] 
then
    echo "Creating folder and downloading model files" 
    wget https://drive.google.com/u/0/uc?id=1MGaKK4nHTd4FWDnvFihb6b8lASjeuN_l&export=download&confirm=t -O $HOME_FOLDER/trained_models.zip
    unzip $HOME_FOLDER/trained_models.zip -d $HOME_FOLDER
fi


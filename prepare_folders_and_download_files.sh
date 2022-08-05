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


HOME_FOLDER="./KDD"
TRAINED_MODEL_FOLDER="./KDD/trained_models/"
if [ ! -d $TRAINED_MODEL_FOLDER ] 
then
    echo "Creating folder and downloading model files" 
    wget https://drive.google.com/u/0/uc?id=1MGaKK4nHTd4FWDnvFihb6b8lASjeuN_l&export=download&confirm=t -O $HOME_FOLDER/trained_models.zip
    unzip $HOME_FOLDER/trained_models.zip -d $HOME_FOLDER
fi


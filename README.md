# Promoter-Recognition-in-Human-DNA-Sequence-Using-Deep-Learning

## Overview
This project aims to recognize promoters in human DNA sequences using deep learning techniques. The project involves converting DNA sequences into spectrogram images and then using Convolutional Neural Networks (CNN) to classify the sequences as promoters or non-promoters.

## Project Structure
The project consists of the following main components:
  - Sequences/: Contains the DNA sequences of promoters and non-promoters.
  - Spectrogram_generation.py: Python code to convert DNA sequences into spectrogram images.
  - Convolutional_Neural_Network.py: Python code to apply CNN on the spectrogram images for classification.

## Dataset
The dataset consists of DNA sequences categorized as promoters and non-promoters:
  - Training set: 4000 sequences (promoters + non-promoters)
  - Testing set: 200 sequences (promoters + non-promoters)

## Workflow
1. **Convert DNA Sequences to Spectrograms:**
     - Run Spectrogram_generation.py to convert the sequences in the Sequences/ folder into spectrogram images.
     - Save the generated spectrograms in the Test/ and Train/ folders respectively.
       
2. **Classify Spectrograms Using CNN:**
     - Run Convolutional_Neural_Network.py to apply the CNN on the spectrogram images.
     - The CNN will classify the spectrograms into promoters and non-promoters.
  
## Results
The output will be the classification of the DNA sequences into promoters and non-promoters based on the spectrogram images.

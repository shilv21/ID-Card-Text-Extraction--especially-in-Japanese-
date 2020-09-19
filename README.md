# ID-Card-Text-Extraction--especially-in-Japanese-
This project developed a method to solve ID card text extraction
# Introduction
  Along with the development of technology, many manual jobs are replacing by machine, especially with AI.
  With this passion, the project tries to solve a problem of automatically ID card text extraction, which will save a lot of time.

# Data preparation
  Collecting data of Japanese characters from a dictionary. 
# Training model
  There are two models in the process. First, the CRAFT model which detect the characters from the images is based on a research paper. 
  The second model which extract the text from thep output of CRAFT model is built on deeplearning.
# Project organization
  The deployed websit will take the image of ID card as the input and return the text as the output. As example below:
  ![Alt text](https://github.com/shilv21/ID-Card-Text-Extraction--especially-in-Japanese-/blob/master/demo.png)

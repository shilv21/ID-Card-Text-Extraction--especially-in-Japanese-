# ID-Card-Text-Extraction--especially-in-Japanese-
This project developed a method to solve ID card text extraction
# Introduction
  Along with the development of technology, many manual jobs are replacing by machine, especially with AI.
  With this passion, the project tries to solve a problem of automatically ID card text extraction, which will save a lot of time.
# How it work
**Model for extract the characters:** The CRAFT model which detect the characters from the images is based on a research paper.

**Model for OCR:** VGG16 is a popular CNN model used for extracting feature vectors. We applied Dropout for a better performance.

**Data preparation:** Collecting data of Japanese characters from a dictionary. 
# Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

## Prerequisites

Make sure you have python 3 and Anaconda installed on your environment.

The installing process is demonstrated in Linux environments.

This project is deployed using Google Cloud Platform (GCP). GCP provides free tier service to host a small web application without paying any fee so feel free to use it to test my project.

GCP also gives you $300 free credits to use within 1 year. That's a lot of money to use for machine learning and professional application deployment service.

[Google Cloud Platform](https://cloud.google.com/)
## Project performance
The deployed website will take the image of ID card as the input and return the text as the output. As example below:
![Alt text](https://github.com/shilv21/ID-Card-Text-Extraction--especially-in-Japanese-/blob/master/demo.png)

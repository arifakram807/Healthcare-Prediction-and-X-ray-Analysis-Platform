
# Healthcare Prediction and X-ray Analysis Platform

Project Overview  
This project integrates machine learning models to predict the hospital length of stay (LOS) for patients, along with an AI-powered image classification tool to analyze chest X-rays. The goal of this platform is to enhance healthcare diagnostics, allowing medical professionals to predict patient outcomes and detect anomalies in X-ray images more efficiently.

Key Features:
Length of Stay Prediction:

Model: Random Forest, trained on a Microsoft healthcare dataset of 100,000 patients.s features such as readmission count, BMI, glucose levels  
Predictors: Include, and engineered features like total issues and health risk scores.   
Results: Achieved a high R² score of 0.94, explaining 94% of the variance in patient length of stay.  
Figure: K-Means Clustering of Patients based on BMI and Glucose.

Figure: Feature Importance in Random Forest Model.

Model Architecture    
Chest X-ray Classification Model   
A Convolutional Neural Network (CNN) built with TensorFlow and Keras.  
The model classifies X-rays into 15 categories, including:  
Atelectasis  
Pneumonia     
Pneumothorax   
Edema   
Cardiomegaly, and more.   
Preprocessing: X-ray images are resized to 300x300, converted to grayscale, and normalized.   
Training: The model was trained for 10 epochs on 256 TFRecord files.

Technologies Used:   
Python: For data preprocessing, feature engineering, and model training.  
TensorFlow: For building the Convolutional Neural Network (CNN) for X-ray image classification.  
Random Forest: Used for the length of stay prediction model.  
Matplotlib: For data visualization of clustering and feature importance.  
Flask: Web framework for building the interactive front-end.  
HTML/CSS: For building the user interface to upload images and display results.


## Screenshots

![App Screenshot](https://i.postimg.cc/0M7NvS64/Home.png)



## Acknowledgements

Dataset
NIH Chest X-ray Dataset: This dataset contains 112,120 frontal-view chest X-ray images from 30,805 unique patients. Images are labeled using Natural Language Processing (NLP) from radiology reports.   
Dataset link: https://www.kaggle.com/datasets/nickuzmenkov/nih-chest-xrays-tfrecords/data   
TFRecord version of the dataset was used for faster training and efficient memory management.  

Patient Length of Stay Dataset: Contains patient health metrics such as BMI, blood urea nitrogen, glucose levels, and others, used for predicting hospital stay duration.  
Dataset link:  https://www.kaggle.com/datasets/aayushchou/hospital-length-of-stay-dataset-microsoft/data


## Future Enhancements


Prediction Improvements: Fine-tune X-ray and length-of-stay models with more advanced architectures (e.g., ResNet or LSTM for sequence prediction).   
Model Explainability: Implement SHAP or LIME for better explainability of the prediction models.   
Integration with Electronic Health Records (EHR): Integrate the platform with real-world EHR systems for live patient data analysis.      

Additional Models: Extend the platform to predict other health-related outcomes such as disease progression or patient recovery rates.
# Semi-Supervised Deep Autoencoder
Design and Implementation of a Robust Machine Learning Architecture for Anomaly Detection and Classification

What is it and how are we doing it?
What is it - what situations does it solve - 
#### Main Goals
- Malware in network traffic or in executable files. The model should be able to accurately classify malware and benign software
- Intrusion Detection: Develop a machine learning model that can detect network intrusions in real-time. The model should be able to classify network traffic as normal or malicious.
- Network Traffic Analysis: Develop a machine learning model that can analyze network traffic to identify potential security threats. The model should be able to identify patterns and anomalies in the network traffic.
- Phishing Detection: Develop a machine learning model that can detect phishing emails. The model should be able to identify emails that are likely to be phishing attempts.

What type of data are we grabbing?
Data from Malware Intrusions, Traffic, Phishing Emails
Useful Links
- Malware - https://www.kaggle.com/c/malware-classification/data?select=dataSample.7z 
- Intrusion Detection - https://research.unsw.edu.au/projects/unsw-nb15-dataset 
- Network Traffic analysis - https://www.unb.ca/cic/datasets/ids-2017.html 



What should the model output?
It should be able to look at data from an intrusion or whatever and decide whether or not the intrusion was malicious or benign



Architecture
Proposed Architecture 

                                  +-----------------+
                                  |   Raw Data      |
                                  +-----------------+
                                           |
                                           |
                                           v
                                  +-----------------+
                                  | Data Preprocessing |
                                  +-----------------+
                                           |
                                           |
                                           v
                        +----------------+----------------+
                        |                 |               |
                        v                 v               v
             +------------------+ +----------------+ +----------------+
             |  Anomaly Detection | | Classification | | Ensembling     |
             +------------------+ +----------------+ +----------------+
                        |                 |               |
                        |                 |               |
                        v                 v               v
             +------------------+ +----------------+ +----------------+
             | Anomalous Patterns| | Predictions    | | Final Output   |
             | Detection Results| | Classification | |                |
             +------------------+ +----------------+ +----------------+


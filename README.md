# 14-813 Course Project 2 – MQTT Dataset

This project involves using the MQTT dataset to conduct data processing and machine learning. This dataset, also known as Message Queuing Telemetry Transport dataset, focuses on predicting the cyber-attacks conducted on an automated house system. Each aspect of the home, from temperature to humidity, is controlled by a sensor under the Internet of Things system and is vulnerable to cyber-attacks from malicious nodes.
This is an extremely large dataset, containing 34 features and 20 million records. Since our computer cannot manage the full volume of the dataset, a small subset of the data, containing 200,000 records was used for the purposes of this assignment. This smaller dataset contains train and test records in the same ratio as the full dataset. A summary of all the columns and the constraints is shown in the table below. 

|Feature	|Constraints	|Description|
|---------|:-------------|:---------------|
|tcp.flags	|Must be string	|TCP flags|
|tcp.time_delta	|Cannot be negative	|Time TCP stream|
|tcp.len	|Cannot be negative	|TCP Segment Len|
|mqtt.conack.flags	|-	|Acknowledge Flags|
|mqtt.conack.flags.reserved	|-	|Reserved|
|mqtt.conack.flags.sp	|-	|Session Present|
|mqtt.conack.val	|-	|Return Code|
|mqtt.conflag.cleansess	|-	|Clean Session Flag|
|mqtt.conflag.passwd	|-	|Password Flag|
|mqtt.conflag.qos	|-	|QoS Level|
|mqtt.conflag.reserved	|-	|(Reserved)|
|mqtt.conflag.retain	|-	|Will Retain|
|mqtt.conflag.uname	|-	|Username Flag|
|mqtt.conflag.willflag	|-	|Will Flag|
|mqtt.conflags	|-	|Connect Flags|
|mqtt.dupflag	|-	|DUP Flag|
|mqtt.hdrflags	|-	|Header Flags|
|mqtt.kalive	|-	|Keep Alive|
|mqtt.len	|Cannot be negative	|Msg Len|
|mqtt.msg	|Cannot be negative	|Message|
|mqtt.msgid	|Cannot be negative	|Message Identifier|
|mqtt.msgtype	|-	|Message Type|
|mqtt.proto_len	|Cannot be negative	|Protocol Name Length|
|mqtt.protoname	|-	|Protocol Name|
|mqtt.qos	|-	|QoS Level|
|mqtt.retain	|-	|Retain|
|mqtt.sub.qos	|-	|Requested QoS|
|mqtt.suback.qos	|-	|Granted QoS|
|mqtt.ver	|-	|Version|
|mqtt.willmsg	|Unique	|Will Message|
|mqtt.willmsg_len	|Cannot be negative	|Will Message Length|
|mqtt.willtopic	|Unique	|Will Topic|
|mqtt.willtopic_len	|Cannot be negative	|Will Topic Length|
|target	|Cannot be null	|Type of Attack|
train_test	|Can only be 0 or 1	|Train and Test Data|

# Final Project

After the MQTT Dataset was ingested and some analytics were carried out, machine learning models were coded and deployed to Google Cloud. 
These tasks were carried out in three Jupyter Notebooks. 

# Notebook 1: TL project -2 Task 1, 2 and 3 till data engineering

This first notebook continues from the first checkpoint and adds the data preprocessing segment to it. Firstly, the dataset was checked for any null values. Then, each column was checked for outliers. If there are too many outliers in a column it will skew the predictions of the model. After carrying out these steps, it was found that this dataset did not contain any null values and only contained columns with at most one outlier. 
Following these steps, column correlation analysis was conducted to drop highly correlated columns as  their presence would increase the computational load of the while not increasing the accuracy much. Through this, four columns were dropped from the dataset. The categorical variables in this dataset were found to be nominal in nature and subsequently one-hot encoded using String Indexer and One Hot Encoder in the Spark pipeline. These variables were assembled with the continuous variables with a Vector Assembler and scaled using Standard Scalar. 

# Notebook 2: Task-3 - Spark-Final and Notebook 3: Task-3 - TF-Local

After the data was preprocessed, it was used as the input to various machine learning models, which can predict if there is an attack and the type of attack. To ease the loading of data in multiple notebooks, the usual pipeline was used, along with modifications to predict for both binary (if there is an attack) and multi-class (type of attack) fitting. For this purpose, two separate pipelines were created, a binary pipeline for predicting an attack and a multi-class pipeline for predicting the type of attack. Under each pipeline, two machine learning models were coded in Spark and in TensorFlow.  Hyper parameter tuning and cross validation were carried out and the best model was then tested against the test dataset. The results are as shown below.

|Type 	|Spark/TensorFlow	|Machine – Learning Model	|Hyper Parameter	|Train Accuracy (%)	|Test Accuracy (%) (BCV)	|Test Accuracy (%) (ACV)	|AUC (BCV)	|AUC (BCV)|
|:-----:|:---------------:|:-----------------------:|:---------------:|:------------------:|:----------------------:|:-----------------------:|:---------:|:-------:|
|Multi – Class	|Spark	|Decision Tree |Max Depth, Max Bins	|81.96	|81.1	|87.03	|-	|-|
|Multi – Class	|Spark	|Logistic Regression	|Reg Parameter, Max Iterations 	|-	|82.85	|82.85	|-	|-|
|Multi – Class	|TensorFlow	|Shallow Neural Network	|NN Width, NN Depth	|82.93	|76.07	|76.53	|-	|-|
|Multi – Class	|TensorFlow |Deep Neural Network	|NN Width, NN Depth	|82.71	|77.12	|79.59	|-	|-|
|Binary	|Spark	|Decision Tree	|Max Depth, Max Bins	|94.08	|94.05	|-	|0.91	|0.95|
|Binary	|Spark |Linear SVC	|Reg Parameter, Max Iterations 	|85.74	|85.33	|-	|0.91	|0.92|
|Binary	|TensorFlow	|Shallow Neural Network	|NN Width, NN Depth	|86.29	|83.71	|85.3	|-	|-|
|Binary	|TensorFlow	|Deep Neural Network	|NN Width, NN Depth	|85.57	|84.69	|84.64	|-	|-|

AUC &#8594; Area under curve (Binary Classification)
BCV &#8594; Before Cross Validation
ACV &#8594; After Cross Validation

# Multi-Class Classification

Under Spark Machine Learning, the decision tree and logistic regression models were used to model the data. Both were classification models capable of conducting multi-class classification. 

While the decision tree model initially gave a train accuracy of 81.96% and a test accuracy of 81.1%, after cross validation was conducted with maximum depth of the tree and maximum bins as the parameters, the test accuracy of the model increased to 87.05%.
The logistic regression model used gave an initial test accuracy of 82.85%. However, even after cross validation using both the regularization parameter and maximum iterations, the test accuracy did not increase at all.

When conducting this in TensorFlow, the neural network was used. A shallow neural network with a single hidden layer and a deep neural network with multiple hidden layers were tested. The shallow neural network gave train and test accuracies of 82.93% and 76.07% respectively. After cross validation using the number of hidden layers and neurons in each layer, it was found that for a shallow neural network, a model containing two hidden layers and 20 neurons per layer is the best, which gave a test accuracy of 76.53%. 

The same analysis was carried out for the deep neural network, which gave train and test accuracies of 82.79% and 74.58% respectively. After conducting cross validation using the same hyper parameters as before, a model containing four hidden layers and 40 neurons per layer is the best, which gave a test accuracy of 76.21%.
From all the machine learning models used for multi-class classification, Spark decision tree gave the highest test accuracy of 87.05%.

# Binary Classification

Under Spark Machine Learning, the decision tree and linear support vector models were used to model the data. Both were classification models used to conduct binary classification. 

The decision tree model initially gave a train accuracy of 94.08%, a test accuracy of 94.05% and the area under the curve as 0.91. After cross validation was conducted with maximum depth of the tree and maximum bins as the parameters, the area under the curve increased to 0.95.

The linear support vector model used gave a train accuracy of 85.74%, a test accuracy of 85.33% and the area under the curve as 0.91. However, after cross validation using both the regularization parameter and maximum iterations, the area under the curve increased to 0.92.

When conducting this in TensorFlow, the neural network was used. A shallow neural network with a single hidden layer and a deep neural network with multiple hidden layers were tested. The shallow neural network gave train and test accuracies of 86.29% and 83.71% respectively. After cross validation using the number of hidden layers and neurons in each layer, it was found that for a shallow neural network, a model containing two hidden layers and 20 neurons per layer is the best, which gave a test accuracy of 85.30%. 

The same analysis was carried out for the deep neural network, which gave train and test accuracies of 85.57% and 84.69% respectively. After conducting cross validation using the same hyper parameters as before, a model containing four hidden layers and 40 neurons per layer is the best, which gave a test accuracy of 84.64%.

From all the machine learning models used for multi-class classification, Spark linear support vector model gave the highest area under the curve as 0.95.

# Deploying Models in Google Cloud

After all the models were tested locally, they were deployed in google cloud. Due to heavy traffic in all the servers, all categorical variables had to be removed to run the files. As a result, the test accuracies of all the models were significantly less as compared to those in the Jupyter notebooks. In addition, a Postgres SQL Instance was created in Cloud and was able to be connected to the Postgres SQL local database through PgAdmin4. 


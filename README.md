# Forecasting with missing data by Fuzzy-based Information Decomposition.

This is a source code for the experiments to demonstrate the forecasting with missing data using the real UAV dataset.

In the source code we have:

- Experiment descriptions.
- The Fuzzy-based Information Decomposition (FID) summary.

## The project

The objective of this project is to evaluate the performance of forcecasting when dealing with missing data resulting from a hacker attack. The attack has compromised some of the data entries, and these entries have been modified from their original values. In such scenarios, restoring the missing data can be a challenging task, especially if the modifications were made with malicious intent.

To address this issue, we employed the Fuzzy-based Information Decomposition (FID) algorithm, which has been previously proposed in [1], to recover the missing data entries. The FID algorithm is a powerful tool that can handle data with uncertain and imprecise information. It decomposes the original data into fuzzy components, each containing a degree of membership that represents the probability of the data belonging to a certain category.

To evaluate the effectiveness of the FID algorithm in restoring the missing data entries, we conducted a series of experiments. The experiments were performed on a dataset containing a significant number of missing entries, resulting from a simulated hacker attack. The algorithm was able to recover a large portion of the missing data entries, providing a reliable and efficient solution for restoring the compromised data.

The details of the FID algorithm and the experimental setup, including the datasets used, the performance metrics, and the results, are provided in the following sections. We also made the source code for the experiments publicly available in a repository for transparency and reproducibility purposes.

In summary, our study demonstrates the potential of the FID algorithm in handling missing data resulting from a hacker attack. The algorithm provides a robust solution that can help restore the compromised data, enabling organizations to make informed decisions based on accurate and reliable data.

## Settings

The experimentation platform utilized Python 3.7.13 and Tensorflow 2.8.0 to conduct the study. Three distinct machine learning models were employed, including the Random Forest Classifier (RF), the K-Neighbors Classifier (KNC), and the Decision Tree Classifier (DT). The RF model was configured with n estimators set to 10 and a random state of 0, while the KNC model was set to n neighbors equaling 5, and the DT model was set with a random state of 0.

## Dataset

Road traffic management is an essential aspect of transportation, especially in urban areas, where traffic congestion can lead to significant problems, such as air pollution and delays. Therefore, developing reliable and efficient traffic management systems is crucial. To achieve this, artificial vision algorithms can be trained to recognize different types of vehicles and their movements in the road network, which can aid in developing efficient traffic management systems.
One critical component of training artificial vision algorithms is access to a diverse dataset with annotated images of vehicles. However, creating such a dataset is often time-consuming and requires considerable resources. To address this issue, a dataset of Spanish road traffic images was developed in this article, obtained from unmanned aerial vehicles (UAV).
The dataset comprises 15,070 images and can be used to train artificial vision algorithms, including those based on convolutional neural networks. The creation process involved multiple stages, including the acquisition of data and images, vehicle labeling, anonymization, and data validation using a simple neural network model. The images were captured using drones, which are similar to those that could be obtained by fixed cameras, in the field of intelligent vehicle management. The dataset provides a wide range of images that capture different types of vehicles in various contexts, such as urban and rural areas, highways, and streets. The presented dataset has the potential to enhance the performance of road traffic vision and management systems. It can aid researchers in developing and testing algorithms that recognize different types of vehicles, their speeds, and movements, which can help optimize traffic flow and reduce congestion.

In summary, the dataset presented can be an invaluable resource for researchers and practitioners working in the field of road traffic management. Its availability can lead to the development of more efficient and reliable traffic management systems, which can significantly improve transportation in urban areas.

Dataset can be downloaded here :https://zenodo.org/record/5776219.

More details of the dataset can be found in the work [2]

## Fuzzy-based Information Decomposition (FID) algorithm

The authors of this study [1] have introduced a novel approach to address two important problems in data analysis: missing data estimation and imbalanced training data. The proposed solution is called the Fuzzy-based Information Decomposition (FID) algorithm, which employs a two-step process to address these issues.

The first step of the algorithm involves weighting, where the contribution of the observed data is quantified using fuzzy membership functions. The second step, recovery, focuses on estimating missing values based on the contribution of the observed data. FID is particularly useful for addressing imbalanced training data by creating synthetic samples for the minority class.

The proposed algorithm is based on the fuzzy set theory, which allows for a more flexible approach to data analysis. To determine the membership degree of recovered data and its contribution weight, FID relies on the minimum and maximum values and the number of recovered data in the discrete universe values.

While the proposed approach has potential for improving classification accuracy, the authors note that there are still areas for improvement. One such area is the consideration of correlation among different column vectors to enhance accuracy. Additionally, as the degree of imbalance and percentage of missing values can vary widely across multi-class datasets, further exploration is needed in this area. Nonetheless, the Fuzzy-based Information Decomposition algorithm is a promising solution for addressing the challenges of missing data estimation and imbalanced training data.

![alt text](https://github.com/ndducnha/uav_awn/FID.png?raw=true)

More details about the FID can be found in the work [1]

## References 

[1] S. Liu, J. Zhang, Y. Xiang and W. Zhou, "Fuzzy-Based Information Decomposition for Incomplete and Imbalanced Data Learning," in IEEE Transactions on Fuzzy Systems, vol. 25, no. 6, pp. 1476-1490, Dec. 2017, doi: 10.1109/TFUZZ.2017.2754998.

[2] Bemposta Rosende, S., Ghisler, S., Fernández-Andrés, J. and Sánchez-Soriano, J., 2022. Dataset: Traffic Images Captured from UAVs for Use in Training Machine Vision Algorithms for Traffic Management. Data, 7(5), p.53.

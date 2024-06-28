# EMERGENCY-VEHICLES-ON-ROAD-NETWORKS-A-NOVEL-GENERATED-DATASET-USING-GANs

This dataset contains 20,000 images of emergency vehicles (ambulances, police cars, fire trucks) collected from real-world scenarios and augmented using standard techniques and GANs. It is designed for training and validating machine learning models, particularly CNNs, to accurately detect emergency vehicles and enhance road safety.

## Citation

Please cite the following research paper when using this dataset:

**Shatnawi, M.; Bani Younes, M. An Enhanced Model for Detecting and Classifying Emergency Vehicles Using a Generative Adversarial Network (GAN). Vehicles 2024, 1, 1–26. https://doi.org/**


## Figures and Tables

### Figure 1: Samples images collected during video recording
![Figure 1](E:/Master/thieses/Slides/sample2.jpg)

Figure 1 displays a selection of images collected during video recording sessions aimed at capturing real-world scenarios of emergency vehicles. These images were meticulously chosen to represent diverse situations and lighting conditions typically encountered on road networks, ensuring the dataset's richness and applicability in training machine learning models.

### Figure 2: GANs Image Generation Process
![Figure 2](<iframe src="https://drive.google.com/file/d/1BGlj5LgEB7clKzHGXgD5xdnSSSzOI62s/preview" width="640" height="480" allow="autoplay"></iframe>)

Figure 2 visually explains the intricate process of image generation using Generative Adversarial Networks (GANs). The generator starts with random noise and progressively refines it through multiple layers, ultimately creating realistic images of emergency vehicles. Meanwhile, the discriminator critically evaluates these generated images to distinguish them from actual data, ensuring the authenticity and quality of the synthetic dataset.

### Table 1: GANs model parameters
| Parameter      | Value              |
|----------------|--------------------|
| latent_dim     | 128                |
| epochs         | 20,000             |
| batch          | 32                 |
| learning rate  | 0.0002             |
| beta_1         | 0.5                |
| dropout_rate   | 0.3                |
| filters        | [64, 128, 128]     |
| kernel_sizes   | [(4, 4), (4, 4), (4, 4)] |
| strides        | [(2, 2), (4, 4), (2, 2)] |

Table 1 provides an overview of the key parameters utilized in the Generative Adversarial Network (GANs) model for generating synthetic images of emergency vehicles. These parameters were carefully selected and tuned to optimize the model's ability to produce diverse and realistic images essential for training robust machine learning algorithms.

### Figure 3: Samples images generated during various epochs
![Figure 3](path/to/figure3.png)

Figure 3 showcases a series of sample images generated at different epochs during the GANs training process. These images illustrate the evolution and refinement of the generated dataset over time, demonstrating the model's learning progression and its capacity to generate increasingly realistic representations of emergency vehicles.

### Table 2: CNN results on final dataset
| Metrics    | Training | Validation | Testing |
|------------|----------|------------|---------|
| Accuracy   | 93.7%    | 91.2%      | 90.9%   |
| Precision  | 96.5%    | 94.2%      | 93.0%   |
| Recall     | 90.4%    | 88.0%      | 88.2%   |
| F1-score   | 93.4%    | 91.0%      | 90.5%   |

Table 2 summarizes the performance metrics of the Convolutional Neural Network (CNN) on the final dataset of emergency vehicle images. These metrics highlight the CNN's effectiveness in accurately classifying emergency vehicles, crucial for enhancing road safety measures through automated detection and response systems.

### Figure 4: Training and Learning Results on Final Dataset
![Figure 4a](path/to/figure4a.png)
![Figure 4b](path/to/figure4b.png)

Figure 4 presents the training and learning results of the CNN model on the final dataset. Subfigure (a) depicts the training learning curve, showing the model's loss reduction and convergence over epochs. Subfigure (b) visualizes the training accuracy learning curve, illustrating the model's improvement in accuracy as it learns to distinguish emergency vehicles from background noise and other objects.

### Table 3: 10-fold cross-validation results on the final dataset
| Eval. Metrics | Fold 1 | Fold 2 | ... | Fold 10 | Mean   |
|---------------|--------|--------|-----|---------|--------|
| Accuracy      | 88.50% | 88.90% | ... | 87.90%  | 87.61% |
| Precision     | 89.20% | 90.10% | ... | 86.90%  | 88.34% |
| Recall        | 86.10% | 87.98% | ... | 84.36%  | 85.41% |
| F1-score      | 87.96% | 88.69% | ... | 86.20%  | 86.84% |

Table 3 presents the results of 10-fold cross-validation performed on the final dataset. This analysis evaluates the CNN model's performance across multiple folds, demonstrating its consistency and reliability in classifying emergency vehicles under varying conditions.

### Figure 5: Improvement in 10-fold cross-validation results before and after augmentation
![Figure 5](path/to/figure5.png)

Figure 5 visualizes the comparative analysis of 10-fold cross-validation results before and after dataset augmentation. It illustrates how augmentation techniques enhanced the model's performance metrics, showcasing the significant improvements in accuracy, precision, recall, and F1-score achieved through increased dataset diversity and robust training methodologies.

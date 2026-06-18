#  CRISP-DM Phase 2: Data Understanding
## Dataset Description

For the development of this project, a custom dataset was created using images captured by the project team members. The purpose of this dataset is to provide the necessary data to train a model capable of identifying drowsiness states through eye analysis.

The dataset is organized into two classes. Class 0 contains images of people with their eyes open or partially open, while Class 1 contains images of people with their eyes closed. This classification makes it possible to distinguish between alert states and potential drowsiness states.

The images were collected under different lighting conditions and with different individuals in order to introduce diversity into the dataset and obtain more representative examples of real-world situations.

## Data Distribution

The dataset consists of a total of 2,000 images distributed equally between the two classes defined for the problem.

| Class | Description | Quantity |
|---------|------------|----------|
| Class 0 | Eyes open or partially open | 1000 |
| Class 1 | Eyes closed | 1000 |
| Total | Complete dataset | 2000 |

The balanced distribution of the classes is an important characteristic because it allows the model to learn both categories equally and reduces the possibility of favoring one class over the other during the learning process.

## Data Variability

During the image collection process, different conditions were considered to introduce variability into the dataset. These include changes in lighting conditions and differences among the individuals photographed.

This diversity is beneficial for model development because it allows the model to learn more general patterns and improves its ability to make predictions on images that were not used during training.

## Data Quality

A general review of the images was conducted to verify that each image was correctly assigned to its corresponding class. In addition, it was confirmed that the eye state could be clearly identified in each image, since this is the main feature used for classification.

Overall, the dataset presents an adequate level of quality for the development of the project, as the images clearly distinguish between open-eye and closed-eye states.

## Initial Findings

From the analysis performed, it was observed that the dataset has a balanced distribution between the two classes and a sufficient number of examples to begin model training. Furthermore, the visual difference between open and closed eyes represents a clearly distinguishable feature, making the application of machine learning techniques for drowsiness detection feasible.

The results obtained during this phase provide a solid foundation for the subsequent stages of the project, including data preparation and model training.
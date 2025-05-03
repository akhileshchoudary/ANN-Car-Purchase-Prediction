# Artificial Neural Networks-Car-Purchase-Prediction

This project leverages an Artificial Neural Network (ANN) to predict the amount a customer might spend on a car, based on key features including Gender, Age, Annual Salary, Credit Card Debt, and Net Worth. Designed as a regression task, this project showcases skills in data preprocessing, neural network modeling, and performance evaluation, making it a valuable addition to my GitHub portfolio.

## Dataset

The dataset used is `Car_Purchasing_Data.csv`, which includes the following customer attributes:

- Customer Name  
- Customer E-mail  
- Country  
- Gender  
- Age  
- Annual Salary  
- Credit Card Debt  
- Net Worth  
- Car Purchase Amount  

For this project, the predictive features are:

- Gender  
- Age  
- Annual Salary  
- Credit Card Debt  
- Net Worth  

**Target Variable**: Car Purchase Amount  

The dataset is included in the repository for ease of use.

## Approach

### Data Preprocessing

- Load the dataset using `pandas`  
- Select relevant features and the target variable  
- Split the data into training, validation, and test sets  

### Model Building

- Construct a sequential ANN using `TensorFlow/Keras`  
- The architecture includes multiple dense layers with appropriate activation functions (details in notebook)  

### Model Training

- Compile the model using a regression-suited optimizer and loss function  
- Train on the training set while validating performance

### 📉 Model Loss Progression

The plot below shows the model's training and validation loss over 20 epochs.

- **Training Loss** decreases steadily, indicating that the model is effectively learning patterns from the training data.
- **Validation Loss** also follows a similar downward trend, showing good generalization to unseen data.
- By the final epoch, validation loss stabilizes around **0.0017**, suggesting minimal overfitting and a well-trained model.

![image](https://github.com/user-attachments/assets/33c7b4b7-7178-4e3d-8c17-7a4c0fa71be0)



### Model Evaluation

- Evaluate performance on the test set using:
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  

### Prediction

- Use the trained model to predict car purchase amounts for new customer data

## Results

The model demonstrated strong predictive performance on the test set:

- **Mean Squared Error (MSE)**: 0.0012  
- **Root Mean Squared Error (RMSE)**: 0.035  
- **R² Score**: 0.95  

These results indicate that predictions deviate by only ~3.5% on average, and the model explains 95% of the variance in the target variable.  
The validation loss reached **~0.0017** by the final epoch, suggesting effective learning with minimal overfitting.  
A plot of training and validation loss progression is included in the notebook.

## Example Prediction

For a customer with:

- Gender: 1 (Male)  
- Age: 30  
- Annual Salary: $100,000  
- Credit Card Debt: $100,985  
- Net Worth: $62,931  

**Predicted Car Purchase Amount**: ~$65,313

## Requirements

- Python 3.x  
- Jupyter Notebook  

### Libraries

- pandas  
- numpy  
- matplotlib  
- seaborn  
- scikit-learn  
- tensorflow  

---


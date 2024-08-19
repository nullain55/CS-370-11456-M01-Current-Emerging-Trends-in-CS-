### Analysis of Accuracy Rates

In this series of experiments, we evaluated three different neural network architectures using the MNIST dataset. The main changes between the models include the number of hidden layers, the activation functions, the introduction of dropout for regularization, and the number of epochs. Here, we compare the results from the three steps and analyze the changes in accuracy rates for the training, validation, and test datasets.

#### Step 1: Defining a simple neural net in Keras
- **Architecture**: 
  - Single Dense layer with softmax activation
- **Epochs**: 200
- **Results**:
  - **Test Accuracy**: 92.45%
  - **Test Score**: 0.256

This simple model acts as a baseline, showing the performance of a minimal network. Given its simplicity, it is expected to have lower accuracy compared to more complex architectures due to underfitting.

#### Step 2: Improving the simple net in Keras with hidden layers
- **Architecture**:
  - Two Dense layers with ReLU activation followed by a softmax output layer
- **Epochs**: 20
- **Results**:
  - **Test Accuracy**: 98.23%
  - **Test Score**: 0.079

Introducing two hidden layers with ReLU activation increases the model's capacity to learn more complex patterns in the data. This model achieves better performance on both training and validation sets. However, without regularization, there might be a risk of overfitting.

#### Step 3: Further improving the simple net in Keras with dropout
- **Architecture**:
  - Two Dense layers with ReLU activation and dropout, followed by a softmax output layer
- **Epochs**: 250
- **Dropout Rate**: 0.3
- **Results**:
  - **Test Accuracy**: 98.47%
  - **Test Score**: 0.072

Adding dropout helps in preventing overfitting by randomly dropping units during training. This regularization technique improves generalization, as seen in the higher validation and test accuracy. The increased number of epochs allows the model to train longer, which can be beneficial when combined with dropout to prevent overfitting.

### Summary of Observations
1. **Training Accuracy**:
   - **Step 1**: The simplest model showed lower training accuracy, indicative of underfitting.
   - **Step 2**: The model with two hidden layers had higher training accuracy due to increased capacity.
   - **Step 3**: The training accuracy was slightly lower than Step 2 due to dropout, which prevents overfitting.

2. **Validation Accuracy**:
   - **Step 1**: Lower due to underfitting.
   - **Step 2**: Improved compared to Step 1, but possibly prone to overfitting.
   - **Step 3**: Highest, showing the benefit of dropout in improving generalization.

3. **Test Accuracy**:
   - **Step 1**: Lowest due to the simplest model architecture.
   - **Step 2**: Higher than Step 1, but might overfit without regularization.
   - **Step 3**: Highest, demonstrating the effectiveness of dropout and a well-balanced architecture.

### Conclusion
The accuracy rates for the training, validation, and test datasets improve as we enhance the model complexity and introduce regularization techniques. The addition of hidden layers allows the network to learn more complex representations, while dropout helps in maintaining generalization by preventing overfitting. The increased number of epochs ensures that the model has enough time to converge. Overall, a balanced approach with adequate regularization and sufficient training time yields the best performance.

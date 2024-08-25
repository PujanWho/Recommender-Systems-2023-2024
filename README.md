# Recommender-System(s) 2023-2024

Recommender System Project 2023-2024  
This project explores the development of personalized and state-of-the-art recommender systems, leveraging collaborative filtering and deep learning techniques.

## Methodology

### Data Source
The dataset used in this project is the MovieLens 100k dataset, accessed through the Python library `Surprise`. This library simplifies the process of building and evaluating recommender systems by providing pre-implemented algorithms and evaluation metrics.

### Recommender System 1 (RS1)

#### Technique
- **Approach:** User-based collaborative filtering with matrix factorization.
- **Algorithm:** Singular Value Decomposition (SVD).
- **Justification for SVD:**
  - Handles sparse user-item matrices effectively, which is crucial for the MovieLens dataset due to the natural sparsity of user ratings.
  - Captures latent factors representing underlying user preferences and item characteristics, leading to more personalized recommendations.
  - Unlike KNN, SVD is better suited for handling the cold start problem.

#### Implementation
- **Library:** Surprise library is used for handling data preprocessing, such as managing sparse data, user-item indexing, and normalizing ratings.
- **Recommendation Generation:** Top N movie recommendations (capped at 10) are generated using the trained SVD model.
- **Evaluation Metrics:** RMSE and NDCG are used to evaluate the accuracy and ranking quality of the recommendations.

### Recommender System 2 (RS2)

#### Technique
- **Approach:** Collaborative filtering enhanced with deep learning techniques.
- **Algorithm:** Stochastic Gradient Descent (SGD) with the Adam optimizer.
- **Justification for SGD with Adam:**
  - **SGD:** More computationally efficient for the size of the dataset and scalable for larger datasets.
  - **Adam Optimizer:** Robust against noisy data and sparse gradients. Adaptive learning rates help avoid local minima, and it reduces sensitivity to hyperparameters, making it ideal for neural networks.

#### Implementation
- **Novelty Metric:** In addition to RMSE and NDCG, the novelty of each recommendation is calculated to assess how uncommon or less popular the recommended items are.

### Evaluation
- **RS1:**
  - **RMSE:** Indicates accuracy in rating predictions.
  - **NDCG:** Measures ranking quality, focusing on the relevance of top recommendations to the user.
- **RS2:**
  - **RMSE and NDCG:** Adjusted for deep learning. While RS1 might show better RMSE, RS2 potentially generalizes better, reducing overfitting.
  - **Novelty:** Provides a balance between popular and less-known recommendations, offering diversity.

### Comparison of RS1 and RS2
- **RS1:** Excels in accuracy (low RMSE) but raises concerns about potential overfitting, as suggested by NDCG.
- **RS2:** Although it has a slightly higher RMSE, RS2’s high NDCG suggests strong ranking performance and better generalization to unseen data, making it more robust for real-world scenarios.

## Running the Recommender Systems

To run the Recommender Systems, follow these steps:

1. **Install Required Libraries:** Ensure you have the `Surprise` library installed.
   ```bash
   pip install scikit-surprise

2. **Run the Recommender System:**
- For RS1 (SVD-based):
  ```
  python rs1.py
  ```
- For RS2 (SGD-based with deep learning):
  ```
  python rs2.py
  ```

3. **User Inputs:**
- **RS1:** Input your `userId` to receive top movie recommendations.
- **RS2:** Similar input process, with additional evaluation for novelty.

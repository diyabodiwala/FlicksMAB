Movie Recommendation System using Multi-Armed Bandits and PyTorch
=====================================================

This project implements a movie recommendation system using multi-armed bandits and PyTorch. The system takes into account the user's past behavior and recommends movies that are likely to engage them.

Features
--------

* Multi-armed bandit algorithm using PyTorch
* Movie embedding using PyTorch
* Recommendation generation using Thompson Sampling
* Evaluation metrics: User engagement and revenue

Dataset
--------

* MovieLens 100K dataset

Getting Started
---------------

### Installation

You can install the required libraries using pip:

pip install -r requirements.txt

### Basic Usage

Here's an example of how to use the movie recommendation system:
```python
from recommender import Recommender

# Load the dataset
data = pd.read_csv('movielens-100k.csv')

# Create a recommender object
recommender = Recommender(data)

# Get recommendations for a user
user_id = 1
recommendations = recommender.get_recommendations(user_id)

print(recommendations)

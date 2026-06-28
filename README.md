# ML Algorithms from Scratch — UT CS Academy 2026

Implementations of core ML algorithms built during the UT Computer 
Science Academy for Machine Learning at UT Austin, June 2026.

## Algorithms Implemented

### Classification (`/classification`)
- **Binary Perceptron** — trained on the Sonar dataset (208 samples, 
  60 features) to classify underwater rocks vs. mines using SGD weight updates
- **Multiclass Perceptron** — 10-class handwritten digit classifier on 
  28×28 pixel images (784 features), one weight vector per class
- **K-Nearest Neighbors** — KNN with Euclidean distance and k-fold 
  cross-validation

### Reinforcement Learning (`/reinforcement_learning`)
- **Q-Learning Agent** — implemented Bellman equation updates, 
  epsilon-greedy exploration, and a nested dictionary Q-table; trained 
  on Berkeley's Gridworld environment

## Results

| Algorithm | Dataset | Result |
|-----------|---------|--------|
| Binary Perceptron | Sonar (208 samples, 60 features) | 82.69% best epoch accuracy |
| Multiclass Perceptron | MNIST digits (1000 training, 10 classes) | 100% training accuracy by epoch 25 |
| Q-Learning | Gridworld (100 episodes) | Converged from negative returns (episode 1) to consistent positive returns (episode 16+) |

## Key Concepts Implemented

- Stochastic gradient descent weight updates
- Multiclass perceptron rule: penalize predicted class weights, 
  reward actual class weights on every mistake
- Bellman equation: `Q(s,a) ← Q(s,a) + α[r + γ·maxQ(s',a') - Q(s,a)]`
- Epsilon-greedy exploration vs. exploitation tradeoff

## What I Wrote vs. What Was Provided

**My implementations:**
- `perceptron_binary.py` — classify function and SGD training loop
- `perceptron_multiclass.py` — PerceptronClassifier with per-class 
  weight vectors and multiclass update rule
- `qlearningAgents.py` — full Q-learning agent including Q-table, 
  Bellman updates, and epsilon-greedy action selection

**Provided framework:** Berkeley Pacman AI infrastructure, 
dataset loaders, and environment simulators

## Program

UT Computer Science Academy for Machine Learning  
University of Texas at Austin — June 2026# ml-algorithms-utcs-2026
KNN, Perceptron, and Q-Learning implementations from UT CS Academy

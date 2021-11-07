# FsHiddenMarkov
Hidden Markov Model for F#. The implementation is based on a paper by Mark Stamp.

A model is defined as $ \lambda = (A, B, \pi) $ with:

- $ A = N x N $ state transition matrix, $ a_{ij} $ is the probability of transitioning from state $ i $ to state $ j $.
- $ B = N x M $ observation probability matrix, $ b_{j}(k) $ being the probability of observing $ k $ in state $ j $.
- $ \pi = N x 1 $ is the initial state distribution vector.

States and observations are simply integers. The mapping to domain types is kept outside of this project.

## Features
1. Compute the probability of an observation sequence, given a model.
2. Predicts the most likely hidden state sequence, given observations.
3. Estimates a Hidden Markov Model given an initialization and observational data.

## References
- [Stamp, Mark - A Revealing Introduction to Hidden Markov Models (2021)](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf)

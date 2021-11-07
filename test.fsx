#r "/home/jovyan/work/FsHiddenMarkov/bin/Debug/net5.0/FsHiddenMarkov.dll"

open System
open FsHiddenMarkov

// The weather example from Mark Stamp's tutorial.
// Used here as a test for verifying the implementation.

// Consider temperatures 0=hot and 1=cold.
// P(hot_t | hot_t-1) = 0.7
// P(cold_t | colt_t-1) = 0.6
// We summarize this information in a state transition probability matrix:
let A = array2D [
    [| 0.7; 0.3 |]
    [| 0.4; 0.6 |]
] 
// The row is which state we are at, the column is which state we transition to.

// Now, we have indirect evidence on what the temperature is/was.
// Suppose the size of tree growth rings are correlated with temperature.
// Consider three sizes: 0=small, 1=medium, 2=large.
// The temperature state is on the rows (hot, cold) and the tree ring sizes are on the columns.
// When it's hot, the probabilities are 0.1, 0.4 and 0.5.
// When it's cold, the probabilities are 0.7, 0.2, 0.1.
// We summarize this information in an observation probability matrix:
let B = array2D [
    [| 0.1; 0.4; 0.5 |]
    [| 0.7; 0.2; 0.1 |]
]

// To get started, we also need some idea on what the initial state is.
// Here, we just pick hot=0.6, cold=0.4.
let pi = [| 0.6; 0.4 |]

// All structures above are row-stochastic, meaning that each element is a 
// probability and the elements of each row sum to 1.

// Now, let's say we have observations:
let O = [| 0; 1; 0; 2 |]

// Determine the most likely state sequence in the HMM sense.

let hmm = HiddenMarkovModel(A, B, pi)

// force float array... > let b = Array2D.zeroCreate<float> N T;;

// Calling manually.
let a, c = alphaPass A B pi O
let b = betaPass A B O c
let gamma, digamma = gammas A B O a b

// Calculated gammas (state probabilities per time step):
// val gamma : float [,] =
//   [[0.1881698098; 0.5194317521; 0.2288776273; 0.8039793969]
//    [0.8118301902; 0.4805682479; 0.7711223727; 0.1960206031]]
// Note that this corresponds to the numbers on page 5 in the tutorial.

// From this, we can tell that the most likely state sequence (in the HMM-sense) is:
// cold -> hot -> cold -> hot
// This should match what we get here:
let predictedStates = hmm.PredictStateSequence O
// And so it is.

// It is interesting to see the probability of the observations.
let logProb = hmm.ObservationProbability O
let prob = exp logProb

// Re-estimate the model, based on these observations.
let hmmRe = hmm.EstimateModel(O)

exp(hmmRe.ObservationProbability O)

// Shows some interesting features.
// Clearly the model overfits completely to the data.
// This wouldn't be a problem if we had more data.

// We have learnt that we can _never_ stay in the same weather.
hmmRe.StateTransitionProbabilities
// We _never_ observe small tree rings in hot weather.
hmmRe.ObservationProbabilities
// We _always_ start in cold weather.
hmmRe.InitialStateDistribution

// In practical applications, we need an ensemble run.
// In practical applications, we might want to just set the initial state distribution
// rather than estimating it like this.
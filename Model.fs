namespace FsHiddenMarkov

open System

/// Suppress warning for "experimental" feature.
/// 3D Array indexing.
#nowarn "57"

[<AutoOpen>]
module Model =

    /// Runs the forward alpha-pass, and returns alphas along with scale factors.
    let alphaPass (An: float[,]) (Bn: float[,]) (pi: float[]) (O: int[]) =
        let N = Array2D.length1 An
        let T = Array.length O
        let c = Array.zeroCreate T
        let a = Array2D.zeroCreate N T

        // Compute a0
        a.[*, 0] <- [| for i in 0..N-1 -> pi.[i] * Bn.[i, O.[0]] |]
        let a0s = a.[*, 0] |> Array.sum
        c.[0] <- 1. / a0s

        // Scale.
        a.[*, 0] <- a.[*, 0] |> Array.map (fun x -> c.[0] * x)

        // Calculate for each timestep.
        for t in 1..T-1 do
            let aits = 
                [| for i in 0..N-1 ->
                    let ait = [| for j in 0..N-1 -> a.[j, t-1]*An.[j, i] |] |> Array.sum
                    ait * Bn.[i, O.[t]]
                |]

            // Scale.
            c.[t] <- 1. / (aits |> Array.sum)
            a.[*, t] <- aits |> Array.map (fun x -> c.[t] * x)

        a, c
        
    /// Runs the backward beta-pass, calculating betas.
    let betaPass (An: float[,]) (Bn: float[,]) (O: int[]) (c: float[])  =
        let N = Array2D.length1 An
        let T = Array.length O
        let b = Array2D.zeroCreate N T

        // Initialize to 1, scaled by ct
        b.[*, T-1] <- Array.create N c.[T-1]

        for t in T-2..-1..0 do
            let bits =
                [| for i in 0..N-1 ->
                    let bit = [| for j in 0..N-1 -> An.[i, j]*Bn.[j, O.[t+1]]*b.[j, t+1] |] |> Array.sum
                    bit * c.[t]
                |]
            b.[*, t] <- bits

        b

    /// Computes and returns gamma(i) and gamma(i, j).
    /// Assumes input alphas and betas are already normalized.
    /// gamma(i, t) = P(xt = qi | O)
    let gammas (An: float[,]) (Bn: float[,]) (O: int[]) (a: float[,]) (b: float[,]) =
        let N = Array2D.length1 An
        let T = Array.length O
        let digamma = Array3D.zeroCreate N N T
        let gamma = Array2D.zeroCreate N T

        for t in 0..T-2 do
            for i in 0..N-1 do
                let gammati = [| for j in 0..N-1 -> a.[i, t]*An.[i, j]*Bn.[j, O.[t+1]]*b.[j, t+1] |]
                digamma.[i, *, t] <- gammati
                gamma.[i, t] <- gammati |> Array.sum
        
        // Special case.
        gamma.[*, T-1] <- a.[*, T-1]

        gamma, digamma

    /// Re-estimates each transition probability.
    /// Returns an N x N matrix.
    let estimateTransitionProbabilities (gamma: float[,]) (digamma: float[,,]) =
        let T = Array2D.length2 gamma
        let N = Array2D.length1 gamma

        let aijs = 
            [ for i in 0..N-1 ->
                // Expected number of transitions from qi to any state.
                let denom = [| for t in 0..T-2 -> gamma.[i, t] |] |> Array.sum
                [| for j in 0..N-1 ->
                    // Expected number of transitions from qi -> qj.
                    let num = [| for t in 0..T-2 -> digamma.[i, j, t] |] |> Array.sum
                    num/denom         
                |]
            ]

        array2D aijs

    /// Given scale factors, calculates log probability of the observations.
    let observationProbability (c: float[]) (O: int[]) =

            // Compute log probabilities.
            let lp = c |> Array.sumBy log
            -lp

    /// Re-estimates observation probabilities, correlation to hidden states.
    /// Returns an N x M matrix.
    let estimateObservationProbabilities (M: int) (gamma: float[,]) (O: int[]) =
        let T = Array2D.length2 gamma
        let N = Array2D.length1 gamma

        let bijs =
            [ for i in 0..N-1 ->
                // Expected number of times the model is in state qj.
                let denom = [| for t in 0..T-1 -> gamma.[i, t] |] |> Array.sum
                [| for k in 0..M-1 -> 
                    // Expected number of times the model is in state qj with observation k.
                    let obsJ = O |> Array.filter (fun o -> o = k)
                    let idx = obsJ |> Array.map (fun o -> O |> Array.findIndex (fun oj -> o = oj )) 
                    let num = idx |> Array.sumBy (fun t -> gamma.[i, t])
                    num/denom
                |]
            ]

        array2D bijs

    type HiddenMarkovModel
        (
            stateTransitionProbabilities: float [,],
            observationProbabilities: float [,],
            initialStateDistribution: float []
        ) =

        do
            if Array2D.length1 stateTransitionProbabilities <> Array2D.length2 stateTransitionProbabilities then
                raise <| ArgumentException("State transition probability matrix must be square.")
            if Array2D.length1 observationProbabilities <> Array2D.length1 stateTransitionProbabilities then
                raise <| ArgumentException("Observation probability matrix must have same length of first dimension as state transition probability matrix.")
            if Array.length initialStateDistribution <> Array2D.length1 stateTransitionProbabilities then
                raise <| ArgumentException("Initial state distribution must have same number of states as state transition probability matrix.")

        let A = stateTransitionProbabilities
        let B = observationProbabilities
        let pi = initialStateDistribution

        let N = Array2D.length1 A
        let M = Array2D.length2 B

        member _.StateTransitionProbabilities = A
        member _.ObservationProbabilities = B
        member _.InitialStateDistribution = pi

        /// Given observations, computes the probability of those observations
        /// according to the model.
        member _.ObservationProbability (O: int[]) =
            let _, c = alphaPass A B pi O
            observationProbability c O

        /// Given observations, computes the most likely state sequence and returns it.
        member _.PredictStateSequence (O: int[]) =
            let T = Array.length O

            let a, c = alphaPass A B pi O
            let b = betaPass A B O c
            let gamma, _ = gammas A B O a b

            // For each time period, find the maximum gamma.
            // From the definition of gamma, this is the most likely state at that time.
            let stateIndices = [|0..T-1|] |> Array.map (fun t -> gamma.[*, t] |> Array.findIndex (fun g -> g = (gamma.[*, t] |> Array.max)))
            stateIndices

        /// Estimates a new model based on existing data and the new observations
        /// Returns the new model.
        member _.EstimateModel(O: int[], ?epsOpt: float, ?maxIterOpt: int) =
            let eps =
                match epsOpt with
                | Some(e) -> e
                | None -> 1e-6
            let maxIter =
                match maxIterOpt with
                | Some(mi) -> mi
                | None -> 1_000_000

            let rec estModel Ax Bx pix lastLikelihood i =
                let a, c = alphaPass Ax Bx pix O
                let b = betaPass Ax Bx O c
                let gamma, digamma = gammas Ax Bx O a b

                // Initial state distribution.
                let pin = gamma.[*, 0]

                let An = estimateTransitionProbabilities gamma digamma
                let Bn = estimateObservationProbabilities M gamma O

                // Check convergence.
                let likelihood = observationProbability c O
                let likelihoodDiff = likelihood - lastLikelihood
                if likelihoodDiff > eps && i < maxIter then
                    estModel An Bn pin likelihood (i+1)
                else
                    HiddenMarkovModel(An, Bn, pin)

            estModel A B pi Double.NegativeInfinity 0
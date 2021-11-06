namespace FsHiddenMarkov

open System

[<AutoOpen>]
module Model =

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

        member _.EstimateModel (obs: int[]) =
            let O = obs.Clone() :?> int[]
            let An = A.Clone() :?> float[,]
            let Bn = B.Clone() :?> float[,]

            let T = Array.length O
            let c = Array.zeroCreate T
            let a = Array2D.zeroCreate N T
            let b = Array2D.zeroCreate N T

            let alphaPass =

                // Compute a0
                a.[*, 0] <- [| for i in 0..N-1 -> pi.[i] * Bn.[i, O.[0]] |]
                let a0s = a.[*, 0] |> Array.sum
                c.[0] <- 1. / a0s

                // Scale.
                a.[*, 0] <- a.[*, 0] |> Array.map (fun x -> c.[0] * x)

                // Calculate for each timestep.
                let rec alphat t =
                    
                    let aits = 
                        [| for i in 0..N-1 ->
                            let ait = [| for j in 0..N-1 -> a.[j, t-1]*An.[j, i] |] |> Array.sum
                            ait * Bn.[i, t]
                        |]

                    // Scale.
                    c.[t] <- 1. / (aits |> Array.sum)
                    a.[*, t] <- aits |> Array.map (fun x -> c.[t] * x)

                    if t < T-1 then
                        alphat (t+1)
                    else
                        ()

                alphat 1 |> ignore

            let betaPass =

                // Let Beta last timestep be 1, scaled by last c.
                b.[*, T-1] <- Array.init N (fun _ -> c.[T-1])


                3





            3.
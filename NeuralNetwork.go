package main

import (
	"math"
	"math/rand"
	"time"
)

const INPUT_LAYER_SIZE = 784
const HIDDEN_LAYER_SIZE = 100
const OUTPUT_LAYER_SIZE = 10
const LEARNING_RATE = 0.035

type NeuralNetwork struct {
	inputToHiddenWeights  [][]float64
	hiddenBias            []float64
	hiddenToOutputWeights [][]float64
	outputBias            []float64
}

func NewNeuralNetwork() (rcvr *NeuralNetwork) {
	rcvr = &NeuralNetwork{}
	rcvr.inputToHiddenWeights = make([][]float64, INPUT_LAYER_SIZE)

	for i := 0; i < INPUT_LAYER_SIZE; i++ {
		rcvr.inputToHiddenWeights[i] = make([]float64, HIDDEN_LAYER_SIZE)
	}
	// inputToHiddenWeights = new double[INPUT_LAYER_SIZE][HIDDEN_LAYER_SIZE];
	rcvr.hiddenBias = make([]float64, HIDDEN_LAYER_SIZE)
	rcvr.hiddenToOutputWeights = make([][]float64, HIDDEN_LAYER_SIZE)
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		rcvr.hiddenToOutputWeights[i] = make([]float64, OUTPUT_LAYER_SIZE)
	}

	rcvr.outputBias = make([]float64, OUTPUT_LAYER_SIZE)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < INPUT_LAYER_SIZE; i++ {
		for j := 0; j < HIDDEN_LAYER_SIZE; j++ {
			rcvr.inputToHiddenWeights[i][j] = rng.Float64() * 0.1
		}
	}
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		rcvr.hiddenBias[i] = rng.Float64() * 0.1
		for j := 0; j < OUTPUT_LAYER_SIZE; j++ {
			rcvr.hiddenToOutputWeights[i][j] = rng.Float64() * 0.1
		}
	}
	for i := 0; i < OUTPUT_LAYER_SIZE; i++ {
		rcvr.outputBias[i] = rng.Float64() * 0.1
	}
	return
}
func (rcvr *NeuralNetwork) Evaluate(data [][]float64, labels []int) float64 {
	correct := 0
	for i := 0; i < len(data); i++ {
		predicted := rcvr.Predict(data[i])
		if predicted == labels[i] {
			correct++
		}
	}
	return float64(correct) / float64(len(data))
}
func (rcvr *NeuralNetwork) Predict(inputs []float64) int {
	hiddenLayerActivations := make([]float64, HIDDEN_LAYER_SIZE)
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < INPUT_LAYER_SIZE; j++ {
			hiddenLayerActivations[i] += rcvr.inputToHiddenWeights[j][i] * inputs[j]
		}
		hiddenLayerActivations[i] += rcvr.hiddenBias[i]
		hiddenLayerActivations[i] = rcvr.sigmoid(hiddenLayerActivations[i])
	}
	outputLayerActivations := make([]float64, OUTPUT_LAYER_SIZE)
	for i := 0; i < OUTPUT_LAYER_SIZE; i++ {
		for j := 0; j < HIDDEN_LAYER_SIZE; j++ {
			outputLayerActivations[i] += rcvr.hiddenToOutputWeights[j][i] * hiddenLayerActivations[j]
		}
		outputLayerActivations[i] += rcvr.outputBias[i]
		outputLayerActivations[i] = rcvr.sigmoid(outputLayerActivations[i])
	}
	maxIndex := 0
	for i := 1; i < OUTPUT_LAYER_SIZE; i++ {
		if outputLayerActivations[i] > outputLayerActivations[maxIndex] {
			maxIndex = i
		}
	}
	return maxIndex
}
func (rcvr *NeuralNetwork) sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
func (rcvr *NeuralNetwork) Train(inputs []float64, target int) {
	hiddenLayerActivations := make([]float64, HIDDEN_LAYER_SIZE)
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < INPUT_LAYER_SIZE; j++ {
			hiddenLayerActivations[i] += rcvr.inputToHiddenWeights[j][i] * inputs[j]
		}
		hiddenLayerActivations[i] += rcvr.hiddenBias[i]
		hiddenLayerActivations[i] = rcvr.sigmoid(hiddenLayerActivations[i])
	}
	outputLayerActivations := make([]float64, OUTPUT_LAYER_SIZE)
	for i := 0; i < OUTPUT_LAYER_SIZE; i++ {
		for j := 0; j < HIDDEN_LAYER_SIZE; j++ {
			outputLayerActivations[i] += rcvr.hiddenToOutputWeights[j][i] * hiddenLayerActivations[j]
		}
		outputLayerActivations[i] += rcvr.outputBias[i]
		outputLayerActivations[i] = rcvr.sigmoid(outputLayerActivations[i])
	}
	outputLayerErrors := make([]float64, OUTPUT_LAYER_SIZE)
	for i := 0; i < OUTPUT_LAYER_SIZE; i++ {

		var suckmyass = 0

		if i == target {
			suckmyass = 1
		}

		outputLayerErrors[i] = outputLayerActivations[i] - float64(suckmyass)
	}
	hiddenLayerErrors := make([]float64, HIDDEN_LAYER_SIZE)
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < OUTPUT_LAYER_SIZE; j++ {
			hiddenLayerErrors[i] += outputLayerErrors[j] * rcvr.hiddenToOutputWeights[i][j]
		}
	}
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		for j := 0; j < OUTPUT_LAYER_SIZE; j++ {
			rcvr.hiddenToOutputWeights[i][j] -= LEARNING_RATE * outputLayerErrors[j] * hiddenLayerActivations[i]
		}
	}
	for i := 0; i < OUTPUT_LAYER_SIZE; i++ {
		rcvr.outputBias[i] -= LEARNING_RATE * outputLayerErrors[i]
	}
	for j := 0; j < INPUT_LAYER_SIZE; j++ {
		for k := 0; k < HIDDEN_LAYER_SIZE; k++ {
			rcvr.inputToHiddenWeights[j][k] -= LEARNING_RATE * hiddenLayerErrors[k] * inputs[j]
		}
	}
	for i := 0; i < HIDDEN_LAYER_SIZE; i++ {
		rcvr.hiddenBias[i] -= LEARNING_RATE * hiddenLayerErrors[i]
	}
}

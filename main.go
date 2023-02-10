package main

import (
	"fmt"
	"path"
)

type Main struct {
}

func NewMain() (rcvr *Main) {
	rcvr = &Main{}
	return
}
func main() {
	nn := NewNeuralNetwork()
	numTrainingExamples := 60000
	numValidationExamples := 10000
	numTestingExamples := 10000
	trainingData := make([][]float64, numTrainingExamples)

	for i := 0; i < numTrainingExamples; i++ {
		trainingData[i] = make([]float64, INPUT_LAYER_SIZE)
	}

	trainingLabels := make([]int, numTrainingExamples)
	validationData := make([][]float64, numValidationExamples)

	for i := 0; i < numValidationExamples; i++ {
		validationData[i] = make([]float64, INPUT_LAYER_SIZE)
	}

	validationLabels := make([]int, numValidationExamples)
	testingData := make([][]float64, numTestingExamples)

	for i := 0; i < numTestingExamples; i++ {
		testingData[i] = make([]float64, INPUT_LAYER_SIZE)
	}

	data, err := NewNumberIdentificationFile(path.Join(".", "data", "mnist_train.csv"))
	if err != nil {
		panic(err)
	}

	for idx, entry := range *data {
		trainingLabels[idx] = entry.expected

		trainingData[idx] = entry.imageData

	}

	data, err = NewNumberIdentificationFile(path.Join(".", "data", "mnist_test.csv"))
	if err != nil {
		panic(err)
	}

	for idx, entry := range *data {
		validationLabels[idx] = entry.expected

		validationData[idx] = entry.imageData
	}

	testingLabels := make([]int, numTestingExamples)

	data, err = NewNumberIdentificationFile(path.Join(".", "data", "mnist_test.csv"))
	if err != nil {
		panic(err)
	}

	for idx, entry := range *data {
		testingLabels[idx] = entry.expected

		testingData[idx] = entry.imageData
	}

	epochs := 40
	patience := 5
	bestEpoch := -1
	bestAccuracy := float64(0)
	for i := 0; i < epochs; i++ {
		for j := 0; j < numTrainingExamples; j++ {
			nn.Train(trainingData[j], trainingLabels[j])
		}
		accuracy := nn.Evaluate(validationData, validationLabels)
		fmt.Println(fmt.Sprintf("%v%v%v%v", "Epoch ", i+1, " accuracy: ", accuracy*100))
		if accuracy > bestAccuracy {
			bestAccuracy = accuracy
			bestEpoch = i
		} else if i-bestEpoch >= patience {
			break
		}
	}
	fmt.Println(fmt.Sprintf("%v%v%v%v", "Best epoch: ", bestEpoch+1, " with accuracy: ", bestAccuracy))
	fmt.Println(fmt.Sprintf("%v%v", "Final test accuracy: ", nn.Evaluate(testingData, testingLabels)))
}

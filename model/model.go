package model

import (
	"math"
	"math/rand"
)

type SimpleModel struct {
	weights1 [][]float64
	weights2 []float64
	bias1    []float64
	bias2    float64
}

func NewSimpleModel() *SimpleModel {
	inputSize := 10
	hiddenSize := 5

	weights1 := make([][]float64, inputSize)
	for i := range weights1 {
		weights1[i] = make([]float64, hiddenSize)
		for j := range weights1[i] {
			weights1[i][j] = rand.Float64()
		}
	}

	weights2 := make([]float64, hiddenSize)
	for i := range weights2 {
		weights2[i] = rand.Float64()
	}

	bias1 := make([]float64, hiddenSize)
	for i := range bias1 {
		bias1[i] = rand.Float64()
	}

	bias2 := rand.Float64()

	return &SimpleModel{
		weights1: weights1,
		weights2: weights2,
		bias1:    bias1,
		bias2:    bias2,
	}
}

func (s *SimpleModel) Train(data []float64) {
	input := data[:len(data)-1]
	output := data[len(data)-1]

	hidden := make([]float64, len(s.bias1))
	for i := range hidden {
		sum := 0.0
		for j := range input {
			sum += input[j] * s.weights1[j][i]
		}
		sum += s.bias1[i]
		hidden[i] = sigmoid(sum)
	}

	prediction := 0.0
	for i := range hidden {
		prediction += hidden[i] * s.weights2[i]
	}
	prediction += s.bias2
	prediction = sigmoid(prediction)

	error := output - prediction

	dOutput := error * prediction * (1 - prediction)

	dHidden := make([]float64, len(hidden))
	for i := range dHidden {
		dHidden[i] = dOutput * s.weights2[i] * hidden[i] * (1 - hidden[i])
	}

	for i := range s.weights2 {
		s.weights2[i] += 0.1 * dOutput * hidden[i]
	}
	s.bias2 += 0.1 * dOutput

	for i := range s.weights1 {
		for j := range s.weights1[i] {
			s.weights1[i][j] += 0.1 * dHidden[j] * input[i]
		}
	}
	for i := range s.bias1 {
		s.bias1[i] += 0.1 * dHidden[i]
	}
}

func (s *SimpleModel) Predict(input []float64) float64 {
	hidden := make([]float64, len(s.bias1))
	for i := range hidden {
		sum := 0.0
		for j := range input {
			sum += input[j] * s.weights1[j][i]
		}
		sum += s.bias1[i]
		hidden[i] = sigmoid(sum)
	}

	prediction := 0.0
	for i := range hidden {
		prediction += hidden[i] * s.weights2[i]
	}
	prediction += s.bias2
	return sigmoid(prediction)
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

package native

import (
	"strings"

	"github.com/destro1t/native_api/native/model"
)

type NativeAPI struct {
	model *model.SimpleModel
}

func NewNativeAPI() *NativeAPI {
	return &NativeAPI{
		model: model.NewSimpleModel(),
	}
}

func (n *NativeAPI) Train(input, output string) {
	inputTokens := strings.Split(input, " ")
	outputTokens := strings.Split(output, " ")

	inputFloats := make([]float64, len(inputTokens))
	for i, token := range inputTokens {
		inputFloats[i] = float64(len(token))
	}

	outputFloat := float64(len(outputTokens))

	n.model.Train(append(inputFloats, outputFloat))
}

func (n *NativeAPI) Predict(input string) string {
	inputTokens := strings.Split(input, " ")
	inputFloats := make([]float64, len(inputTokens))
	for i, token := range inputTokens {
		inputFloats[i] = float64(len(token))
	}

	prediction := n.model.Predict(inputFloats)

	if prediction > 0.5 {
		return "Russian"
	} else {
		return "Russian"
	}
}

# [file name]: API_DOCS.md
# Native API Documentation

## Overview
Native 1.0 Experimental is a simple neural network implementation for text classification. It supports basic training and prediction operations.

## Quick Start

```go
package main

import (
	"api/native"
	"fmt"
)

func main() {
	api := native.NewNativeAPI()
	
	// Training
	api.Train("Hello world", "1")  // English
	api.Train("Привет мир", "0")   // Russian
	
	// Prediction
	fmt.Println(api.Predict("Test sentence")) // Output: English
}

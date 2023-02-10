package main

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

type numberIdentificationElement struct {
	expected  int
	imageData []float64
}

func (n *numberIdentificationElement) GetExpectedValues(outputLen int) []float64 {
	expected := []float64{}

	for i := 0; i < outputLen; i++ {
		if i+1 == n.expected {
			expected = append(expected, 1)
		} else {
			expected = append(expected, 0)
		}
	}

	return expected
}

type NumberIdentificationFile []numberIdentificationElement

func NewNumberIdentificationFile(csvFile string) (*NumberIdentificationFile, error) {
	f, err := os.Open(csvFile)
	if err != nil {
		return nil, err
	}

	entries := NumberIdentificationFile{}

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	scanner.Scan() // Skip label line since we don't use that
	for scanner.Scan() {
		line := scanner.Text()
		items := strings.Split(line, ",")

		if len(items) < 1 {
			return nil, fmt.Errorf("invalid entry in CSV file")
		}
		expected, err := strconv.Atoi(items[0])
		if err != nil {
			return nil, err
		}

		imageData := []float64{}
		for i := 1; i < len(items); i++ {
			value, err := strconv.Atoi(items[i])
			if err != nil {
				return nil, err
			}
			imageData = append(imageData, float64(value)/255.0)
		}

		entries = append(entries, numberIdentificationElement{
			expected:  expected,
			imageData: imageData,
		})
	}

	return &entries, nil
}

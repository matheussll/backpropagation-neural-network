import Layer from './Layer';

class Network {
  constructor(numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers, numberOfOutputNeurons, learningRate, regularizationValue) {
    const hiddenLayers = [];
    for (let i = 0; i < numberOfHiddenLayers; i += 1) {
      if (!hiddenLayers.length) {
        const hiddenLayer = new Layer(numberOfInputs + 1, numberOfHiddenNeurons);
        hiddenLayers.push(hiddenLayer);
      } else {
        const hiddenLayer = new Layer(numberOfHiddenNeurons + 1, numberOfHiddenNeurons);
        hiddenLayers.push(hiddenLayer);
      }
    }
    this.hiddenLayers = hiddenLayers;
    this.outputLayer = new Layer(numberOfHiddenNeurons + 1, numberOfOutputNeurons, true);
    this.inputs = [];
    this.learningRate = learningRate;
    this.regularizationValue = regularizationValue;
  }

  forwardPropagate(inputs) {
    this.inputs = inputs;
    this.hiddenLayers[0].neurons.forEach((neuron) => {
      neuron.inputs = inputs;
    });

    this.hiddenLayers.forEach((layer, index) => {
      const outputs = [];

      layer.neurons.forEach((neuron, neuronIndex) => {
        if (neuronIndex) {
          neuron.activate(neuron.inputs);
        }
        outputs.push(neuron.output);
      });
      if (this.hiddenLayers[index + 1]) {
        this.hiddenLayers[index + 1].neurons.forEach((neuron) => {
          neuron.inputs = outputs;
        });
      } else {
        this.outputLayer.neurons.forEach((neuron) => {
          neuron.inputs = outputs;
          neuron.activate(neuron.inputs);
          console.log('output: ', neuron.output);
        });
      }
    });
  }

  backwardsErrorPropagation(expectedOutput) {
    this.outputLayer.neurons.forEach((neuron, neuronIndex) => {
      neuron.error = (neuron.output - expectedOutput[neuronIndex]);
      // console.log('network error: ', neuron.outputDerivative);
    });

    this.hiddenLayers.slice().reverse().forEach((layer, index) => {
      if (!index) {
        layer.neurons.forEach((neuron, neuronIndex) => {
          if (neuronIndex) {
            let errorSum = 0;
            this.outputLayer.neurons.forEach((outputLayerNeuron) => {
              errorSum += outputLayerNeuron.weights[neuronIndex] * outputLayerNeuron.error;
            });
            const error = errorSum * neuron.outputDerivative;
            neuron.error = error;

             console.log('Neuronio ', neuronIndex, 'da camada de output - ', neuron);
          }
        });
      } else {
        layer.neurons.forEach((neuron, neuronIndex) => {
          if (neuronIndex) {
            let errorSum = 0;
            this.hiddenLayers.slice().reverse()[index - 1].neurons.forEach((previousLayerNeuron) => {
              errorSum += previousLayerNeuron.weights[neuronIndex] * previousLayerNeuron.error;
            });
            const error = errorSum * neuron.outputDerivative;
            neuron.error = error;
            // console.log('Neuronio ', neuronIndex, 'da camada ', index, ' - ', neuron);
          }
        });
      }
    });
  }

  calculateGradientsAndUpdateWeights() {
    this.outputLayer.neurons.forEach((outputLayerNeuron) => {
      // console.log('OUTPUT NEURON BEFORE: ', outputLayerNeuron.weights);
      const gradients = [];
      this.hiddenLayers.slice().reverse()[0].neurons.forEach((neuron, neuronIndex) => {
        let gradient = neuron.output * outputLayerNeuron.error;
        gradient += this.regularizationValue * outputLayerNeuron.weights[neuronIndex];
        gradients.push(gradient);
      });
      outputLayerNeuron.weightGradients = gradients;
      const newWeights = [];
      outputLayerNeuron.weights.forEach((weight, weightIndex) => {
        weight -= outputLayerNeuron.weightGradients[weightIndex] * this.learningRate;
        newWeights.push(weight);
      });
      outputLayerNeuron.weights = newWeights;
      // console.log('OUTPUT NEURON ', outputLayerNeuron.error);
      // console.log('===================');
    });

    this.hiddenLayers.slice().reverse().forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron) => {
        // console.log('NEURON ', neuronIndex, ' OF LAYER ', layerIndex, 'BEFORE: ', neuron.weights);
        const gradients = [];
        if (this.hiddenLayers[layerIndex + 1]) {
          this.hiddenLayers[layerIndex + 1].neurons.forEach((nextLayerNeuron, nextLayerNeuronIndex) => {
            let gradient = nextLayerNeuron.output * neuron.error;
            if (nextLayerNeuronIndex) {
              gradient += this.regularizationValue * neuron.weights[nextLayerNeuronIndex];
            }
            gradients.push(gradient);
          });
        } else {
          this.inputs.forEach((input, inputIndex) => {
            let gradient = input * neuron.error;
            if (inputIndex) {
              gradient += this.regularizationValue * neuron.weights[inputIndex];
            }
            gradients.push(gradient);
          });
        }
        neuron.weightGradients = gradients;
        const newWeights = [];

        neuron.weights.forEach((weight, weightIndex) => {
          weight -= neuron.weightGradients[weightIndex] * this.learningRate;
          newWeights.push(weight);
        });
        neuron.weights = newWeights;
        // console.log('NEURON ', neuronIndex, ' OF LAYER ', layerIndex, 'GRADIENTS: ', neuron.weightGradients, ' , ERROR: ', neuron.error);
        // console.log('===================');
      });
    });
  }

  calculateCostFunction(expectedOutput) {
    let sum = 0;
    this.outputLayer.neurons.forEach((neuron, neuronIndex) => {
      sum += (neuron.output - expectedOutput[neuronIndex]) ** 2;
    });
    return sum / (2 * this.outputLayer.neurons.length);
  }

  calculateCostFunction2(expectedOutput) {
    let sum = 0;
    this.outputLayer.neurons.forEach((neuron, neuronIndex) => {
      const e = 10 ** -8;
      let cost = -expectedOutput[neuronIndex] * Math.log(neuron.output + e);
      // console.log('esperado: ', expectedOutput[neuronIndex], 'obtido: ', neuron.output);
      cost += -(1 - expectedOutput[neuronIndex]) * Math.log((1 - neuron.output) + e);
      sum += cost;
    });
    return sum;
  }

  regularization(trainingSetLength) {
    let sum = 0;
    this.outputLayer.neurons.forEach((neuron) => {
      neuron.weights.forEach((weight) => {
        sum += weight ** 2;
      });
    });

    this.hiddenLayers.forEach((layer) => {
      layer.neurons.forEach((neuron) => {
        neuron.weights.forEach((weight) => {
          sum += weight ** 2;
        });
      });
    });

    sum *= this.regularizationValue;
    sum /= (2 * trainingSetLength);
    return sum;
  }

  train(trainingSet) {
    trainingSet.forEach((item) => {
      item.input.unshift(1);
    });
    for (let i = 0; i < 20000; i += 1) {
      let sum = 0;
      trainingSet.forEach((item) => {
        this.forwardPropagate(item.input);
        this.backwardsErrorPropagation(item.output);
        this.calculateGradientsAndUpdateWeights();
        const cost = this.calculateCostFunction(item.output);
        // console.log('custo do input ', j, ': ', cost);
        sum += cost;
      });
      sum /= trainingSet.length;
      sum += this.regularization(trainingSet.length);
      // console.log('custo final: ', sum);
    }
  }

  predict(input) {
    input.unshift(1);
    this.forwardPropagate(input);
    const outputs = [];
    this.outputLayer.neurons.forEach((neuron) => {
      outputs.push(neuron.output);
    });
    console.log('Outputs: ', outputs);
  }
}

export default Network;
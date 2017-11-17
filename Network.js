import Layer from './Layer';

class Network {
  constructor(numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers, numberOfOutputNeurons, learningRate) {
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
  }

  forwardPropagate(inputs) {
    inputs.unshift(1);
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
        // console.log('Neuronio ', neuronIndex, 'da camada ', index, ' - ', neuron);

        outputs.push(neuron.output);
      });
      if (this.hiddenLayers[index + 1]) {
        this.hiddenLayers[index + 1].neurons.forEach((neuron) => {
          neuron.inputs = outputs;
        });
      } else {
        this.outputLayer.neurons.forEach((neuron, neuronIndex) => {
          neuron.inputs = outputs;
          neuron.activate(neuron.inputs);
          // console.log('Neuronio ', neuronIndex, 'da camada de output - ', neuron);
        });
      }
    });
  }

  backwardsErrorPropagation(expectedOutput) {
    this.outputLayer.neurons.forEach((neuron) => {
      neuron.error = neuron.output - expectedOutput;
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
            // console.log('Neuronio ', neuronIndex, 'da camada de output - ', neuron);
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
      console.log('OUTPUT NEURON BEFORE: ', outputLayerNeuron);
      const gradients = [];
      this.hiddenLayers.slice().reverse()[0].neurons.forEach((neuron) => {
        gradients.push(neuron.output * outputLayerNeuron.error);
      });
      outputLayerNeuron.weightGradients = gradients;
      const newWeights = [];
      outputLayerNeuron.weights.forEach((weight, weightIndex) => {
        weight -= outputLayerNeuron.weightGradients[weightIndex] * this.learningRate;
        newWeights.push(weight);
      });
      outputLayerNeuron.weights = newWeights;
      console.log('OUTPUT NEURON AFTER: ', outputLayerNeuron);
      console.log('===================');
    });

    this.hiddenLayers.slice().reverse().forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        console.log('NEURON ', neuronIndex, ' OF LAYER ', layerIndex, 'BEFORE: ', neuron);
        const gradients = [];
        if (this.hiddenLayers[layerIndex + 1]) {
          this.hiddenLayers[layerIndex + 1].neurons.forEach((nextLayerNeuron) => {
            gradients.push(nextLayerNeuron.output * neuron.error);
          });
        } else {
          this.inputs.forEach((input) => {
            gradients.push(input * neuron.error);
          });
        }
        neuron.weightGradients = gradients;
        const newWeights = [];

        neuron.weights.forEach((weight, weightIndex) => {
          weight -= neuron.weightGradients[weightIndex] * this.learningRate;
          newWeights.push(weight);
        });
        neuron.weights = newWeights;
        console.log('NEURON ', neuronIndex, ' OF LAYER ', layerIndex, 'AFTER: ', neuron);
        console.log('===================');
      });
    });
  }
}

export default Network;

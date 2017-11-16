import Layer from './Layer';

class Network {
  constructor(numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers, numberOfOutputNeurons) {
    const hiddenLayers = [];
    for (let i = 0; i < numberOfHiddenNeurons; i += 1) {
      if (!hiddenLayers.length) {
        const hiddenLayer = new Layer(numberOfInputs, numberOfHiddenNeurons);
        hiddenLayers.push(hiddenLayer);
      } else {
        const hiddenLayer = new Layer(numberOfHiddenNeurons, numberOfHiddenNeurons);
        hiddenLayers.push(hiddenLayer);
      }
    }
    this.hiddenLayers = hiddenLayers;
    this.outputLayer = new Layer(numberOfHiddenNeurons, numberOfOutputNeurons);
  }

  forwardPropagate(inputs) {
    this.hiddenLayers[0].neurons.forEach((neuron) => {
      neuron.inputs = inputs;
    });

    this.hiddenLayers.forEach((layer, index) => {
      const outputs = [];

      layer.neurons.forEach((neuron) => {
        neuron.activate(neuron.inputs);
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
          let errorSum = 0;
          this.outputLayer.neurons.forEach((outputLayerNeuron) => {
            errorSum += outputLayerNeuron.weights[neuronIndex] * outputLayerNeuron.error;
          });
          const error = errorSum * neuron.outputDerivative;
          neuron.error = error;
        });
      } else {
        layer.neurons.forEach((neuron, neuronIndex) => {
          let errorSum = 0;
          this.hiddenLayers.slice().reverse()[index - 1].neurons.forEach((previousLayerNeuron) => {
            errorSum += previousLayerNeuron.weights[neuronIndex] * previousLayerNeuron.error;
          });
          console.log(errorSum);
          const error = errorSum * neuron.outputDerivative;
          neuron.error = error;
        });
      }
    });
  }
}

export default Network;

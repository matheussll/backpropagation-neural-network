import Neuron from './Neuron2';

class Layer {
  constructor(numberOfInputs, numberOfNeurons) {
    this.neurons = [];
    for (let i = 0; i < numberOfNeurons; i += 1) {
      const neuron = new Neuron();
      const neuronWeights = [];
      for (let j = 0; j < numberOfInputs; j += 1) {
        neuronWeights.push(Math.random());
      }
      neuron.bias = Math.random();
      neuron.weights = neuronWeights;
      this.neurons.push(neuron);
    }
    this.outputs = [];
  }
}

export default Layer;

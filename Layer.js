import Neuron from './Neuron';

class Layer {
  constructor(numberOfInputs, numberOfNeurons, isOutputLayer) {
    this.neurons = [];
    const amountOfNeurons = isOutputLayer ? numberOfNeurons : numberOfNeurons + 1;

    for (let i = 0; i < amountOfNeurons; i += 1) {
      const neuron = new Neuron();
      const neuronWeights = [];

      if (i === 0 && !isOutputLayer) {
        neuron.output = 0;
        neuron.outputDerivative = 0;
      }
      for (let j = 0; j < numberOfInputs; j += 1) {
        neuronWeights.push(Math.random());
      }
      neuron.weights = neuronWeights;
      this.neurons.push(neuron);
    }
    this.outputs = [];
  }
}

export default Layer;

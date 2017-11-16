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
        // console.log('neuronio da camada atual -', index, neuron);

        // console.log('neuron output: ', neuron.output);
        outputs.push(neuron.output);
        // console.log('NEURONIO DA CAMADA ', index, neuron);
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

    // this.hiddenLayers.slice().reverse().forEach((layer, index) => {
    //   if (!index) {

    //   }
    // });
  }
}

export default Network;

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
    // console.log(this);
    this.inputs = inputs;
    this.hiddenLayers[0].neurons.forEach((neuron) => {
      neuron.inputs = inputs;
    });
    //this.hiddenLayers[0].neurons[1].weights = [0, 0.39];/////////////////////////////////////////////
    //this.hiddenLayers[1].neurons[1].weights = [0, 0.94];/////////////////////////////////////////////
    
    this.hiddenLayers.forEach((layer, index) => {
      const outputs = [];

      layer.neurons.forEach((neuron, neuronIndex) => {
        if (neuronIndex) {
          //console.log('neuron', neuronIndex, 'da camada', index);/////////////////////////////////////////////          
          neuron.activate(neuron.inputs);
          // console.log('neuron', neuronIndex, 'da camada', index, 'tem ativação = ', neuron.output);/////////////////////////////////////////////
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
          // console.log(neuron.inputs);
          //this.outputLayer.neurons[0].weights = [0, 0.55];/////////////////////////////////////////////          
          neuron.activate(neuron.inputs);
          // console.log(neuron.output);
          console.log('neuron',  'da ult camada', 'tem ativação = ', neuron.output);/////////////////////////////////////////////
        });
      }
    });
  }

  backwardsErrorPropagation(expectedOutput) {
    this.outputLayer.neurons.forEach((neuron, neuronIndex) => {
      neuron.error = (neuron.output - expectedOutput[neuronIndex]);
      console.log('neuron',  'da ult camada', 'tem erro = ', neuron.error);/////////////////////////////////////////////
      
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
            // console.log('neuron', neuronIndex, 'da camada', index, 'tem erro = ', neuron.error);/////////////////////////////////////////////
            
            
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
            //console.log('neuron', neuronIndex, 'da camada', index, 'tem erro = ', neuron.error);/////////////////////////////////////////////
            
          }
        });
      }
    });
  }

  calculateGradientsAndUpdateWeights() {
    this.outputLayer.neurons.forEach((outputLayerNeuron) => {
      const gradients = [];
      this.hiddenLayers.slice().reverse()[0].neurons.forEach((neuron, neuronIndex) => {
        let gradient = neuron.output * outputLayerNeuron.error;
        
        gradient += this.regularizationValue * outputLayerNeuron.weights[neuronIndex];
        //console.log('Gradiente da conexão N1_Output:', gradient,  '=', neuron.output, '*', outputLayerNeuron.error);  /////////////////////////////////
        gradients.push(gradient);
      });
      outputLayerNeuron.weightGradients = gradients;
      const newWeights = [];
      outputLayerNeuron.weights.forEach((weight, weightIndex) => {
        weight -= outputLayerNeuron.weightGradients[weightIndex] * this.learningRate;
        newWeights.push(weight);
      });
      outputLayerNeuron.weights = newWeights;
    });

    this.hiddenLayers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron) => {
        const gradients = [];
        if (!layerIndex) {
          this.inputs.forEach((input, inputIndex) => {
            let gradient = input * neuron.error;
            if (inputIndex) {
              gradient += this.regularizationValue * neuron.weights[inputIndex];
              //console.log('Gradiente da conexão Input_N1:', gradient,  '=', input, '*', neuron.error);  /////////////////////////////////
              
            }
            gradients.push(gradient);
          });
        } else {
          this.hiddenLayers[layerIndex - 1].neurons.forEach((previousLayerNeuron, previousLayerNeuronIndex) => {
            let gradient = previousLayerNeuron.output * neuron.error;
            if (previousLayerNeuronIndex) {
              gradient += this.regularizationValue * neuron.weights[previousLayerNeuronIndex];
              //console.log('Gradiente da conexão N1_N2:', gradient,  '=', previousLayerNeuron.output, '*', neuron.error);  /////////////////////////////////              
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
      item.input.unshift(1);/////////////////////////////////////////////

    });
    for (let i = 0; i < 5000; i += 1) {
      let sum = 0;
      trainingSet.forEach((item) => {
        //console.log('mamaçao', item.input); /////////////////////////////////////////////
        this.forwardPropagate(item.input);
        this.backwardsErrorPropagation(item.output);
        this.calculateGradientsAndUpdateWeights();
        const cost = this.calculateCostFunction(item.output);
        sum += cost;
        //console.log('=================================================================================');
      });
      sum /= trainingSet.length;
      sum += this.regularization(trainingSet.length);
      //console.log('custo final: ', sum);
    }
  }

  predict(input) {
    input.unshift(1);
    this.forwardPropagate(input);
    const outputs = [];
    this.outputLayer.neurons.forEach((neuron) => {
      outputs.push(neuron.output);
    });
    //console.log('Outputs: ', outputs);
  }
}

export default Network;

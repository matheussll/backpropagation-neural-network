const transfer = value => (1 / (1 + Math.exp(-value))); // sigmoid
const transferDerivative = value => transfer(value) * (1 - transfer(value));

class Neuron {
  constructor(weights) {
    this.weights = weights;
    this.output = 0;
    this.outputDerivative = 0;
    this.inputs = [];
    this.error = 0;
    this.weightGradients = [];
  }

  activate(inputs) {
    let activationValue = 0;
    //console.log('inputs ', inputs);  /////////////////////////////////////////////
    //console.log('weights ', this.weights);  /////////////////////////////////////////////

    inputs.forEach((input, index) => {
      activationValue += input * this.weights[index];
    });
    this.output = transfer(activationValue);
    this.outputDerivative = transferDerivative(activationValue);
  }
}

export default Neuron;

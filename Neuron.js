const transfer = value => (1 / (1 + Math.exp(-value))); // sigmoid
const transferDerivative = value => transfer(value) * (1 - transfer(value));

class Neuron {
  constructor(weights, bias) {
    this.weights = weights;
    this.bias = bias;
    this.output = 0;
    this.outputDerivative = 0;
    this.inputs = [];
    this.error = 0;
    this.weightGradients = [];
  }

  activate(inputs) {
    let activationValue = this.bias;
    inputs.forEach((input, index) => {
      activationValue += input * this.weights[index];
    });
    this.output = transfer(activationValue);
    this.outputDerivative = transferDerivative(activationValue);
  }
}

export default Neuron;

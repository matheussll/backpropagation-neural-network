const transfer = value => (1 / (1 + Math.exp(-value))); // sigmoid
const transferDerivative = value => transfer(value) * (1 - transfer(value));

// const heaviside = value => (value >= 0 ? 1 : 0);

class Neuron {
  constructor(weights, bias) {
    this.weights = weights;
    this.bias = bias;
    this.output = 0;
    this.inputs = [];
    this.error = 0;
  }

  activate(inputs) {
    // console.log('inputs: ', inputs);
    let activationValue = this.bias;
    inputs.forEach((input, index) => {
      activationValue += input * this.weights[index];
    });
    this.output = transfer(activationValue);
    // console.log(this.output);
  }
}

export default Neuron;

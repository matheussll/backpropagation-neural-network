const sigmoid = value => (1 / (1 + Math.exp(-value)));
const heaviside = value => (value >= 0 ? 1 : 0);
const sigmoidThreshold = value => (value >= 0.5 ? 1 : 0);
// const sigmoidDerivative = value => sigmoid(value) * (1 - sigmoid(value));

class Neuron {
  constructor(bias, interactions, learnRate) {
    this.bias = bias;
    this.interactions = interactions;
    this.learnRate = learnRate;
    this.weights = [];
    this.biasWeight = 0;
  }

  initRandomWeights(numberOfInputs) {
    for (let i = 0; i < numberOfInputs; i += 1) {
      this.weights.push(Math.random());
    }
    this.biasWeight = Math.random();
  }

  train(trainingSet) {
    this.initRandomWeights(trainingSet[0].input.length);
  }

  trainWithoutInteractions(trainingSet) {
    this.initRandomWeights(trainingSet[0].input.length);
    let diff = 1;
    let i = 0;
    while (diff) {
      diff = this.calculateDifferences(trainingSet);
      i++;
    }
  }

  calculateDifferences(trainingSet) {
    let diff = 0;
    trainingSet.forEach((entry) => {
      // console.log(entry);
      const result = this.run(entry.input);
      // console.log('Result: ', result);
      if (result !== entry.output) {
        diff = entry.output - result;
        // error = true;
        this.adjustWeights(diff, entry.input);
      }
    });
    console.log('diff: ', diff);
    return diff;
  }

  adjustWeights(error, inputs) {
    this.weights.forEach((weight, index) => {
      this.weights[index] += (error * inputs[index] * this.learnRate);
    });
    this.biasWeight += (error * this.bias * this.learnRate)
  }

  run(inputs) {
    let weightSum = 0;
    inputs.forEach((input, index) => {
      weightSum += (input * this.weights[index]);
    });
    weightSum += (this.bias * this.biasWeight);
    // console.log('Bias Weigth: ', this.biasWeight);
    // console.log('weightSum: ', weightSum);
    // return heaviside(weightSum);
    return sigmoidThreshold(sigmoid(weightSum));
  }

  predict(inputs) {
    console.log('Output will be: ', this.run(inputs));
  }
}

export default Neuron;

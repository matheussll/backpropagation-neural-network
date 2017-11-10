const sigmoid = value => (1 / (1 + Math.exp(-value)));
const heaviside = value => (value >= 0 ? 1 : 0);

class Node {
  constructor(bias, interactions, learnRate) {
    this.bias = bias;
    this.interactions = interactions;
    this.learnRate = learnRate;
    this.weights = [];
  }

  initRandomWeights(numberOfInputs) {
    for (let i = 0; i < numberOfInputs; i += 1) {
      this.weights.push(Math.random());
    }
  }

  train(trainingSet) {
    this.initRandomWeights(trainingSet[0].input.length);
  }

  trainWithoutInteractions(trainingSet) {
    this.initRandomWeights(trainingSet[0].input.length);
    let error = true;
    let i = 0;

    while (error) {
      error = false;
      trainingSet.forEach((entry, index) => {
        const result = this.run(entry.input);
        // console.log('TEST:', entry, result);
        if (result !== entry.output) {
          const diff = entry.output - result;
          error = true;
          this.adjustweights(diff, entry.input);
        }
      });
    // console.log(`Error:  + ${error}`);
    }
  }

  adjustweights(error, inputs) {
    this.weights.forEach((weight, index) => {
      this.weights[index] += (error * inputs[index] * this.learnRate);
    });
  }

  run(inputs) {
    let weightSum = 0;
    inputs.forEach((input, index) => {
      weightSum += (input * this.weights[index]);
    });
    weightSum += this.bias;
    // console.log(weightSum);
    return heaviside(weightSum);
  }

  predict(inputs) {
    console.log('Output will be: ', this.run(inputs));
  }
}

export default Node;

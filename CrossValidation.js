const createCrossValidationsSets = (dataSet) => {
  const validationSet = JSON.parse(JSON.stringify(dataSet));
  const trainingSet = validationSet.splice(0, validationSet.length * 0.8);
  return { trainingSet, validationSet };
};

const dataSetToFolds = (dataSet) => {
  const folds = [];
  const dataSetCopy = JSON.parse(JSON.stringify(dataSet));
  while (dataSetCopy.length) {
    folds.push(dataSetCopy.splice(0, dataSet.length / 5));
  }
  return folds;
};

const shuffleArray = (array) => {
  const arrayToShuffle = JSON.parse(JSON.stringify(array));
  for (let i = arrayToShuffle.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    [arrayToShuffle[i], arrayToShuffle[j]] = [arrayToShuffle[j], arrayToShuffle[i]];
  }
  return arrayToShuffle;
};

const createNetworkTestSets = (folds) => {
  const sets = [];
  for (let i = 0; i < folds.length; i += 1) {
    let trainingSet = [];
    let validationSet;
    folds.forEach((fold, foldIndex) => {
      if (foldIndex !== i) {
        trainingSet = trainingSet.concat(fold);
      } else {
        validationSet = fold;
      }
    });
    sets.push({ validationSet, trainingSet });
  }
  return sets;
};

class CrossValidation {
  constructor(trainingSet, networks) {
    if (trainingSet.length < 5) {
      throw 'Número insuficiente de entradas de treinamento';
    }
    this.trainingSet = trainingSet;
    this.accuracy = 0;
    this.recall = 0;
    this.precision = 0;
    this.error = 0;
    const shuffledArray = shuffleArray(trainingSet);
    const sets = createCrossValidationsSets(shuffledArray);
    const folds = dataSetToFolds(sets.trainingSet);
    this.validationSet = sets.validationSet;
    this.crossValidationSets = createNetworkTestSets(folds);
    this.networks = networks;
  }

  train() {
    this.networks.forEach((network, networkIndex) => {
      console.log(this.crossValidationSets[networkIndex].validationSet.length);
      network.train(this.crossValidationSets[networkIndex].trainingSet);
      console.log('=================================');
    });
    this.test();
    // this.networks[1].train(this.trainingSet);
  }

  test() {
    this.networks.forEach((network, networkIndex) => {
      const outputs = [];
      this.crossValidationSets[networkIndex].validationSet.forEach((entry) => {
        const output = network.predict(entry.input);
        // console.log(output);
        const i = output.indexOf(Math.max(...output));
        const outputNormalized = output.map((value, index) => (index === i ? 1 : 0));
        console.log('output obtido: ', outputNormalized, 'output esperado: ', entry.output, 'rede numero: ', networkIndex);

        outputs.push(outputNormalized);
      });
      // console.log(outputs);
    });
  }

  // calculatePerformance() {
  //   const vp = 0;
  //   const vn = 0;
  //   const fp = 0;
  //   const fn = 0;

  //   const predictedValue = trainingSet.outputNeurons;
  //   this.trainingSet.output.forEach(expectedValue, (index) => {
  //     if (0) { // expectedValue > 0.5 && predictedValue > 0.5
  //       // expects positive, predicts positive
  //       vp++;
  //     } else if (1) { // expectedValue < 0.5 && predictedValue > 0.5
  //       //  expects negative, predicts positive
  //       fp++;
  //     } else if (2) { // expectedValue > 0.5 && predictedValue < 0.5
  //       //  expects positive, predicts negative
  //       fn++;
  //     } else if (3) { // expectedValue < 0.5 && predictedValue < 0.5
  //       //  expects negative, predicts negative
  //       vn++;
  //     }
  //   });
  //   const n = vp + vn + fp + fn;

  //   this.accuracy = (vp + fn) / n;
  //   this.recall = (vp) / (vp + fn);
  //   this.precision = (vp) / (vp + fp);
  // }
}

export default CrossValidation;

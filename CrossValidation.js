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
      //console.log(this.crossValidationSets[networkIndex].validationSet.length);
      network.train(this.crossValidationSets[networkIndex].trainingSet);
      //console.log('=================================');
    });
    this.test();
    // this.networks[1].train(this.trainingSet);
  }

  test() {
    this.networks.forEach((network, networkIndex) => {
      const outputs = [];
      const expectedOutputs = [];
      this.crossValidationSets[networkIndex].validationSet.forEach((entry) => {
        const output = network.predict(entry.input);
        // console.log(output);
        const i = output.indexOf(Math.max(...output));
        const outputNormalized = output.map((value, index) => (index === i ? 1 : 0));
        //console.log('output obtido: ', outputNormalized, 'output esperado: ', entry.output, 'rede numero: ', networkIndex);

        outputs.push(outputNormalized);
        expectedOutputs.push(entry.output);

      });

      const classes = [
        { output: [0, 0, 1] },
        { output: [0, 1, 0] },
        { output: [1, 0, 0] }];

      classes.forEach((positiveClass, index) => {
        //const parameters = [{vp: 0, vn: 0, fp: 0, fn: 0}];
        const parameters = this.calculatePerformance(outputs, expectedOutputs, index, parameters, positiveClass.output);
        //console.log('metricas de desempenho: ', parameters);
      });
      console.log('=================================');
      // console.log(outputs);
    });
  }

  calculatePerformance(outputNormalized, output, index, parameters, positiveClass) {
    let vp = 0;
    let vn = 0;
    let fp = 0;
    let fn = 0;

    for(let i = 0; i < outputNormalized.length; i++){
      let pred = outputNormalized[i];
      let out = output[i];

      if((pred[0] == out[0] 
        && pred[1] == out[1] 
        && pred[2] == out[2])
        && pred[0] == positiveClass[0] 
        && pred[1] == positiveClass[1] 
        && pred[2] == positiveClass[2]){
          vp++;
      }
      else if((pred[0] != out[0] 
        || pred[1] != out[1] 
        || pred[2] != out[2]) 
        && pred[0] == positiveClass[0] 
        && pred[1] == positiveClass[1] 
        && pred[2] == positiveClass[2]){
          fp++;
      }
      else if((pred[0] != out[0]
        || pred[1] != out[1] 
        || pred[2] != out[2]) 
        && (pred[0] != positiveClass[0] 
        || pred[1] != positiveClass[1] 
        || pred[2] != positiveClass[2])){
          fn++;
      }
      else if((pred[0] == out[0]
        && pred[1] == out[1] 
        && pred[2] == out[2]) 
        && (pred[0] != positiveClass[0] 
        || pred[1] != positiveClass[1] 
        || pred[2] != positiveClass[2])){
          vn++;
      }
    }

    const n = vp + vn + fp + fn;

    let accuracy = (vp + fn) / n;
    let recall = (vp) / (vp + fn);
    let precision = (vp) / (vp + fp);
    
    if((vp + fn) == 0){
      let recall = 0;
    }
    if((vp + fp) == 0){
      let precision = 0;
    }
    if(vp == 0){
      let recall = 0;
      let precision = 0;
    }

    console.log('acc: ', accuracy, 'rec: ', recall, 'prec: ', precision);
    return {index, vp, fp, fn, vn};
  }
}

export default CrossValidation;

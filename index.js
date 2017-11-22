import Network from './Network';
import TrainingSet from './TrainingSet';

const normalizeMinMax = (min, max, val) => {
  if (max === min) {
    min = 0;
  }
  const delta = max - min;
  return delta ? ((val - min) / delta) : 0;
};

const normalize = (normalizationValues) => {
  const trainingSetToNormalize = TrainingSet;
  trainingSetToNormalize.forEach((item) => {
    const newInputs = [];
    item.input.forEach((value, index) => {
      newInputs.push(normalizeMinMax(normalizationValues.min[index], normalizationValues.max[index], value));
    });
    item.input = newInputs;
  });
  return trainingSetToNormalize;
};

const normalizationValues = (trainingSet) => {
  const separatedArray = [];
  trainingSet[0].input.forEach(() => {
    separatedArray.push([]);
  });

  trainingSet.forEach((item) => {
    item.input.forEach((input, index) => {
      separatedArray[index].push(input);
    });
  });
  const min = [];
  const max = [];
  separatedArray.forEach((array) => {
    min.push(Math.min(...array));
    max.push(Math.max(...array));
  });
  return { min, max };
};

const trainingSetNormalized = normalize(normalizationValues(TrainingSet)); /////////////////////////////////////////////

/* network parameters - numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers,
 numberOfOutputNeurons, learningRate, regularizationValue */
const network = new Network(TrainingSet[0].input.length, 2, 2, TrainingSet[0].output.length, 0.7, 0);

console.log(trainingSetNormalized); /////////////////////////////////////////////
network.train(trainingSetNormalized);
//network.train(TrainingSet); /////////////////////////////////////////////

trainingSetNormalized.forEach((item) => {
//TrainingSet.forEach((item) => {
 // network.predict(item.output);
});

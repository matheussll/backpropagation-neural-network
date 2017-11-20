import Network from './Network';
import TrainingSet from './XOR';



// const trainingSet = [
//   { input: [25, 670], output: 0 },
//   { input: [29, 665], output: 0 },
//   { input: [32, 610], output: 0 },
//   { input: [40, 650], output: 0 },
//   { input: [48, 500], output: 1 },
//   { input: [25, 730], output: 1 },
//   { input: [28, 805], output: 1 },
//   { input: [35, 740], output: 1 },
//   { input: [41, 710], output: 1 },
//   { input: [45, 660], output: 1 },
//   { input: [48, 580], output: 1 },
//   { input: [55, 540], output: 1 },
// ];

const trainingSet = [
  { input: [0.25, 0.270], output: [0, 1] },
  { input: [0.29, 0.365], output: [0, 1] },
  { input: [0.32, 0.510], output: [0, 0] },
  { input: [0.40, 0.850], output: [0, 0] },
  { input: [0.48, 0.500], output: [1, 1] },
  { input: [0.25, 0.730], output: [1, 0] },
  { input: [0.28, 0.805], output: [1, 1] },
  { input: [0.35, 0.740], output: [1, 0] },
  { input: [0.41, 0.710], output: [1, 1] },
  { input: [0.45, 0.660], output: [1, 0] },
  { input: [0.48, 0.580], output: [1, 0] },
  { input: [0.15, 0.140], output: [1, 1] },
];

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

const a = TrainingSet;
const trainingSetNormalized = normalize(normalizationValues(TrainingSet));

/* network parameters - numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers,
 numberOfOutputNeurons, learningRate, regularizationValue */


const network = new Network(trainingSetNormalized[0].input.length, 2, 2, trainingSetNormalized[0].output.length, 0.2, 0);

network.hiddenLayers[0].neurons[1].weights = [0.36449279348403785, 0.056698228708220055, 0.962149304287057];
network.hiddenLayers[0].neurons[2].weights = [0.7609116098510047, 0.47408399602275386, 0.28207868902918354];

network.hiddenLayers[1].neurons[1].weights = [0.06363118331889916, 0.1501699060755619, 0.4933759734192569];
network.hiddenLayers[1].neurons[2].weights = [0.6611255020002691, 0.11955570868044108, 0.42568609774091004];

network.outputLayer.neurons[0].weights = [0.3265604394066415, 0.8352813420165219, 0.4122636456119215];

network.train(trainingSetNormalized);

trainingSetNormalized.forEach((item) => {
  network.predict(item.output);
});

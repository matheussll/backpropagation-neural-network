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
  const delta = max - min;
  return ((val - min) / delta);
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

const trainingSetNormalized = normalize(normalizationValues(TrainingSet));

console.log(trainingSetNormalized[0].output.length);

/* network parameters - numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers,
 numberOfOutputNeurons, learningRate, regularizationValue */

const network = new Network(trainingSetNormalized[0].input.length, 3, 8, trainingSetNormalized[0].output.length, 0.4, 0.2);

network.train(trainingSetNormalized);

trainingSetNormalized.forEach((item) => {
  network.predict(item.output);
});

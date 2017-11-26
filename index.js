import Network from './Network';
import TrainingSet from './TrainingSet1';
import CrossValidation from './CrossValidation';

// import TrainingSet from './TrainingSet<index_of_training_set>'

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

const trainingSetNormalized = normalize(normalizationValues(TrainingSet));

/* network parameters - numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers,
 numberOfOutputNeurons, learningRate, regularizationValue */

const networks = [];
// const networksParamsToTest = [
//   { hiddenNeurons: 50, hiddenLayers: 1, learningRate: 0.8, regularizationValue: 0.0002 },
//   { hiddenNeurons: 1, hiddenLayers: 1, learningRate: 0.5, regularizationValue: 0.0001 },
//   { hiddenNeurons: 2, hiddenLayers: 2, learningRate: 0.5, regularizationValue: 0.002 },
//   { hiddenNeurons: 3, hiddenLayers: 2, learningRate: 0.2, regularizationValue: 0.002 },
//   { hiddenNeurons: 5, hiddenLayers: 1, learningRate: 0.9, regularizationValue: 0.0002 },
// ];

// const networksParamsToTest = [
//   { hiddenNeurons: 2, hiddenLayers: 1, learningRate: 0.5, regularizationValue: 0.3 },
//   { hiddenNeurons: 2, hiddenLayers: 3, learningRate: 0.4, regularizationValue: 0.1 },
//   { hiddenNeurons: 5, hiddenLayers: 3, learningRate: 0.5, regularizationValue: 0.2 },
//   { hiddenNeurons: 2, hiddenLayers: 2, learningRate: 0.5, regularizationValue: 0.3 },
//   { hiddenNeurons: 5, hiddenLayers: 2, learningRate: 0.5, regularizationValue: 0.2 },
// ];

const networksParamsToTest = [
  { hiddenNeurons: 5, hiddenLayers: 3, learningRate: 0.5, regularizationValue: 0.2 },
  { hiddenNeurons: 2, hiddenLayers: 2, learningRate: 0.8, regularizationValue: 0.2 },
  { hiddenNeurons: 2, hiddenLayers: 1, learningRate: 0.7, regularizationValue: 0.02 },
  { hiddenNeurons: 2, hiddenLayers: 1, learningRate: 0.4, regularizationValue: 0.2 },
  { hiddenNeurons: 5, hiddenLayers: 1, learningRate: 0.9, regularizationValue: 0.02 },
];

// const networksParamsToTest = [
//   { hiddenNeurons: 1, hiddenLayers: 3, learningRate: 0.5, regularizationValue: 0.01},
//   { hiddenNeurons: 2, hiddenLayers: 2, learningRate: 0.1, regularizationValue: 0.005 },
//   { hiddenNeurons: 2, hiddenLayers: 4, learningRate: 0.3, regularizationValue: 0.2 },
//   { hiddenNeurons: 8, hiddenLayers: 1, learningRate: 0.2, regularizationValue: 0.1 },
//   { hiddenNeurons: 3, hiddenLayers: 6, learningRate: 0.4, regularizationValue: 0.03 },
// ];

networksParamsToTest.forEach((param) => {
  const { hiddenNeurons, hiddenLayers, learningRate, regularizationValue } = param;
  const network = new Network(trainingSetNormalized[0].input.length, hiddenNeurons, hiddenLayers, trainingSetNormalized[0].output.length, learningRate, regularizationValue);
  networks.push(network);
});

const crossValidation = new CrossValidation(trainingSetNormalized, networks);

//const network = new Network(trainingSetNormalized[0].input.length, networksParamsToTest[0].hiddenNeurons, networksParamsToTest[0].hiddenLayers, trainingSetNormalized[0].output.length, networksParamsToTest[0].learningRate, networksParamsToTest[0].regularizationValue);
// network.train(trainingSetNormalized);

crossValidation.train();

// To get the chosen network's performance parameters
//crossValidation.train2();

//
// const errors = [];
// crossValidation.validationSet.forEach((entry) => {
//   const output = network.predict(entry.input);
//   // console.log(output);
//   const i = output.indexOf(Math.max(...output));
//   const outputNormalized = output.map((value, index) => (index === i ? 1 : 0));
//   console.log('output obtido: ', outputNormalized, 'output esperado: ', entry.output);
//   errors.push(network.calculateCostFunction(entry.output));

// });



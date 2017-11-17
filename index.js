import Network from './Network';

/* network parameters - numberOfInputs, numberOfHiddenNeurons, numberOfHiddenLayers,
 numberOfOutputNeurons, learningRate */
const network = new Network(2, 3, 3, 1, 0.2);

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
  { input: [0.25, 0.270], output: 0 },
  { input: [0.29, 0.365], output: 0 },
  { input: [0.32, 0.510], output: 0 },
  { input: [0.40, 0.850], output: 0 },
  { input: [0.48, 0.500], output: 1 },
  { input: [0.25, 0.730], output: 1 },
  { input: [0.28, 0.805], output: 1 },
  { input: [0.35, 0.740], output: 1 },
  { input: [0.41, 0.710], output: 1 },
  { input: [0.45, 0.660], output: 1 },
  { input: [0.48, 0.580], output: 1 },
  { input: [0.55, 0.540], output: 1 },
];

network.train(trainingSet, 10);
network.predict([0.25, 0.270, 1]);

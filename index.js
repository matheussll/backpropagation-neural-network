import Neuron from './Neuron';

const network = new Neuron(1, 3000, 0.2);

const trainingSet = [
  { input: [10, 100], output: 0 },
  { input: [15, 90], output: 0 },
  { input: [15, 40], output: 0 },
  { input: [20, 25], output: 1 },
  { input: [25, 30], output: 1 },
  { input: [50, 10], output: 1 },
];

const inputToPredict = [20, 20];

network.trainWithoutInteractions(trainingSet);
network.predict(inputToPredict);


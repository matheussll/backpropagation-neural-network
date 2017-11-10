import Neuron from './Neuron';

const network = new Neuron(1, 3000, 0.2);

const trainingSet = [
  { input: [25, 670], output: 0 },
  { input: [29, 665], output: 0 },
  { input: [32, 610], output: 0 },
  { input: [40, 650], output: 0 },
  { input: [48, 500], output: 1 },
  { input: [25, 730], output: 1 },
  { input: [28, 805], output: 1 },
  { input: [35, 740], output: 1 },
  { input: [41, 710], output: 1 },
  { input: [45, 660], output: 1 },
  { input: [48, 580], output: 1 },
  { input: [55, 540], output: 1 },
];

const inputToPredict = [35, 740];

network.trainWithoutInteractions(trainingSet);
network.predict(inputToPredict);


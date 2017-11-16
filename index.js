import Neuron from './Neuron';
import Network from './Network';

const network = new Network(2, 3, 3, 1);

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


// console.log('Before activation: ');
// console.log('output ', network.hiddenLayers[2].neurons[0]);

network.forwardPropagate([0, 1]);
// console.log('After activation: ');
// console.log('output ', network.hiddenLayers[2].neurons[0]);

network.backwardsErrorPropagation(1);
console.log('After backward propagation: ');
console.log('output ', network.hiddenLayers[2].neurons[0]);
console.log('output ', network.hiddenLayers[1].neurons[1]);
console.log('output ', network.hiddenLayers[0].neurons[2]);


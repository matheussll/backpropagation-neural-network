import Network from './Network';

const w = [];
const gradients = [];
const network = new Network(1, 2, 1, 1, 0.1, 0);
const traininSet = [{ input: [0.81], output: [1] }];
const epslon = 0.000005;

const weights = network.setupTestNetwork();

w[0] = weights[0];
w[1] = weights[1];
w[2] = weights[2];
w[3] = weights[3];

network.hiddenLayers[0].neurons[0].output = 0;
gradients[0] = network.hiddenLayers[0].neurons[1].weightGradients[1];
gradients[1] = network.hiddenLayers[0].neurons[2].weightGradients[1];
gradients[2] = network.outputLayer.neurons[0].weightGradients[1];
gradients[3] = network.outputLayer.neurons[0].weightGradients[2];

const modifiedNetworks = [];
const errorPlusEpslon = [];
const errorMinusEpslon = [];
traininSet[0].input.unshift(0);

for (let i = 0; i < 8; i++) {
  modifiedNetworks[i] = new Network(1, 2, 1, 1, 0.1, 0);
  modifiedNetworks[i].hiddenLayers.forEach((layer) => {
    layer.neurons[0].output = 0;
  });
}

// w0


modifiedNetworks[0].hiddenLayers[0].neurons[1].weights[1] = w[0] + epslon;
modifiedNetworks[0].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[0].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[0].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[0].forwardPropagate(traininSet[0].input);
modifiedNetworks[0].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[0].calculateGradientsAndUpdateWeights();
errorPlusEpslon[0] = calculateCostFunction3(modifiedNetworks[0], traininSet[0].output);


modifiedNetworks[1].hiddenLayers[0].neurons[1].weights[1] = w[0] - epslon;
modifiedNetworks[1].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[1].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[1].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[1].forwardPropagate(traininSet[0].input);
modifiedNetworks[1].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[1].calculateGradientsAndUpdateWeights();
errorPlusEpslon[1] = calculateCostFunction3(modifiedNetworks[1], traininSet[0].output);

// w0

// w1

modifiedNetworks[2].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[2].hiddenLayers[0].neurons[2].weights[1] = w[1] + epslon;
modifiedNetworks[2].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[2].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[2].forwardPropagate(traininSet[0].input);
modifiedNetworks[2].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[2].calculateGradientsAndUpdateWeights();
errorPlusEpslon[2] = calculateCostFunction3(modifiedNetworks[2], traininSet[0].output);


modifiedNetworks[3].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[3].hiddenLayers[0].neurons[2].weights[1] = w[1] - epslon;
modifiedNetworks[3].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[3].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[3].forwardPropagate(traininSet[0].input);
modifiedNetworks[3].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[3].calculateGradientsAndUpdateWeights();
errorPlusEpslon[3] = calculateCostFunction3(modifiedNetworks[3], traininSet[0].output);

// w1

// w2

modifiedNetworks[4].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[4].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[4].outputLayer.neurons[0].weights[1] = w[2] + epslon;
modifiedNetworks[4].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[4].forwardPropagate(traininSet[0].input);
modifiedNetworks[4].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[4].calculateGradientsAndUpdateWeights();
errorPlusEpslon[4] = calculateCostFunction3(modifiedNetworks[4], traininSet[0].output);


modifiedNetworks[5].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[5].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[5].outputLayer.neurons[0].weights[1] = w[2] - epslon;
modifiedNetworks[5].outputLayer.neurons[0].weights[2] = w[3];

modifiedNetworks[5].forwardPropagate(traininSet[0].input);
modifiedNetworks[5].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[5].calculateGradientsAndUpdateWeights();
errorPlusEpslon[5] = calculateCostFunction3(modifiedNetworks[5], traininSet[0].output);

// w2

// w3

modifiedNetworks[6].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[6].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[6].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[6].outputLayer.neurons[0].weights[2] = w[3] + epslon;

modifiedNetworks[6].forwardPropagate(traininSet[0].input);
modifiedNetworks[6].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[6].calculateGradientsAndUpdateWeights();
errorPlusEpslon[6] = calculateCostFunction3(modifiedNetworks[6], traininSet[0].output);


modifiedNetworks[7].hiddenLayers[0].neurons[1].weights[1] = w[0];
modifiedNetworks[7].hiddenLayers[0].neurons[2].weights[1] = w[1];
modifiedNetworks[7].outputLayer.neurons[0].weights[1] = w[2];
modifiedNetworks[7].outputLayer.neurons[0].weights[2] = w[3] - epslon;

modifiedNetworks[7].forwardPropagate(traininSet[0].input);
modifiedNetworks[7].backwardsErrorPropagation(traininSet[0].output);
modifiedNetworks[7].calculateGradientsAndUpdateWeights();
errorPlusEpslon[7] = calculateCostFunction3(modifiedNetworks[7], traininSet[0].output);

// w3

const numericGradients = [];
numericGradients[0] = (errorPlusEpslon[0] - errorPlusEpslon[1]) / (2 * epslon);
numericGradients[1] = (errorPlusEpslon[2] - errorPlusEpslon[3]) / (2 * epslon);
numericGradients[2] = (errorPlusEpslon[4] - errorPlusEpslon[5]) / (2 * epslon);
numericGradients[3] = (errorPlusEpslon[6] - errorPlusEpslon[7]) / (2 * epslon);

const diff = [];
diff[0] = numericGradients[0] - gradients[0];
diff[1] = numericGradients[1] - gradients[1];
diff[2] = numericGradients[2] - gradients[2];
diff[3] = numericGradients[3] - gradients[3];

console.log('================= Gradient Checking =================');
console.log('Index: ', 0, ' - Numeric Gradient: ', numericGradients[0], ' - Gradient: ', gradients[0], ' - Diff: ', diff[0]);
console.log('Index: ', 1, ' - Numeric Gradient: ', numericGradients[1], ' - Gradient: ', gradients[1], ' - Diff: ', diff[1]);
console.log('Index: ', 2, ' - Numeric Gradient: ', numericGradients[2], ' - Gradient: ', gradients[2], ' - Diff: ', diff[2]);
console.log('Index: ', 3, ' - Numeric Gradient: ', numericGradients[3], ' - Gradient: ', gradients[3], ' - Diff: ', diff[3]);

function calculateCostFunction3(network, expectedOutput) {
  let cost = -expectedOutput * Math.log(network.outputLayer.neurons[0].output);
  cost += -(1 - expectedOutput) * Math.log((1 - network.outputLayer.neurons[0].output));
  return cost;
};
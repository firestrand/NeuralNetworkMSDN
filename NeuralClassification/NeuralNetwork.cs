using System;
using System.Linq;

namespace NeuralClassification
{
    class NeuralNetwork
    {
        private readonly int _numInput;
        private readonly int _numHidden;
        private readonly int _numOutput;

        private readonly double[] _inputs;
        private readonly double[][] _ihWeights; // input-to-hidden
        private readonly double[] _ihSums;
        private readonly double[] _ihBiases;
        private readonly double[] _ihOutputs;
        private readonly double[][] _hoWeights;  // hidden-to-output
        private readonly double[] _hoSums;
        private readonly double[] _hoBiases;
        private readonly double[] _outputs;

        static Random _rnd;
        private const double Epsilon = 1e-10;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            _numInput = numInput;
            _numHidden = numHidden;
            _numOutput = numOutput;

            _inputs = new double[numInput];
            _ihWeights = Helpers.MakeMatrix(numInput, numHidden);
            _ihSums = new double[numHidden];
            _ihBiases = new double[numHidden];
            _ihOutputs = new double[numHidden];
            _hoWeights = Helpers.MakeMatrix(numHidden, numOutput);
            _hoSums = new double[numOutput];
            _hoBiases = new double[numOutput];
            _outputs = new double[numOutput];

            _rnd = new Random(0);
        }

        public void SetWeights(double[] weights)
        {
            int numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            if (weights.Length != numWeights)
                throw new Exception("The weights array length: " + weights.Length + " does not match the total number of weights and biases: " + numWeights);

            int k = 0; // points into weights param

            for (int i = 0; i < _numInput; ++i)
                for (int j = 0; j < _numHidden; ++j)
                    _ihWeights[i][j] = weights[k++];

            for (int i = 0; i < _numHidden; ++i)
                _ihBiases[i] = weights[k++];

            for (int i = 0; i < _numHidden; ++i)
                for (int j = 0; j < _numOutput; ++j)
                    _hoWeights[i][j] = weights[k++];

            for (int i = 0; i < _numOutput; ++i)
                _hoBiases[i] = weights[k++];
        }

        public double[] ComputeOutputs(double[] currInputs)
        {
            if (_inputs.Length != _numInput)
                throw new Exception("Inputs array length " + _inputs.Length + " does not match NN numInput value " + _numInput);

            for (int i = 0; i < _numHidden; ++i)
                _ihSums[i] = 0.0;
            //for (int i = 0; i < numHidden; ++i)
            //  this.ihOutputs[i] = 0.0;
            for (int i = 0; i < _numOutput; ++i)
                _hoSums[i] = 0.0;
            //for (int i = 0; i < numOutput; ++i)
            //  this.outputs[i] = 0.0;


            for (int i = 0; i < currInputs.Length; ++i) // copy
                _inputs[i] = currInputs[i];

            //Console.WriteLine("Inputs:");
            //ShowVector(this.inputs);

            //Console.WriteLine("input-to-hidden weights:");
            //ShowMatrix(this.ihWeights);

            for (int j = 0; j < _numHidden; ++j)  // compute input-to-hidden sums
                for (int i = 0; i < _numInput; ++i)
                    _ihSums[j] += _inputs[i] * _ihWeights[i][j];

            //Console.WriteLine("input-to-hidden sums:");
            //ShowVector(this.ihSums);

            //Console.WriteLine("input-to-hidden biases:");
            //ShowVector(ihBiases);

            for (int i = 0; i < _numHidden; ++i)  // add biases to input-to-hidden sums
                _ihSums[i] += _ihBiases[i];

            //Console.WriteLine("input-to-hidden sums after adding biases:");
            //ShowVector(this.ihSums);

            for (int i = 0; i < _numHidden; ++i)   // determine input-to-hidden output
                //ihOutputs[i] = StepFunction(ihSums[i]); // step function
                _ihOutputs[i] = SigmoidFunction(_ihSums[i]);
            //ihOutputs[i] = TanhFunction(ihSums[i]);

            //Console.WriteLine("input-to-hidden outputs after sigmoid:");
            //ShowVector(this.ihOutputs);

            //Console.WriteLine("hidden-to-output weights:");
            //ShowMatrix(hoWeights);


            for (int j = 0; j < _numOutput; ++j)   // compute hidden-to-output sums
                for (int i = 0; i < _numHidden; ++i)
                    _hoSums[j] += _ihOutputs[i] * _hoWeights[i][j];

            //Console.WriteLine("hidden-to-output sums:");
            //ShowVector(hoSums);

            //Console.WriteLine("hidden-to-output biases:");
            //ShowVector(this.hoBiases);

            for (int i = 0; i < _numOutput; ++i)  // add biases to input-to-hidden sums
                _hoSums[i] += _hoBiases[i];

            //Console.WriteLine("hidden-to-output sums after adding biases:");
            //ShowVector(this.hoSums);

            //for (int i = 0; i < numOutput; ++i)   // determine hidden-to-output result
            //  this.outputs[i] = SigmoidFunction(hoSums[i]);  // step function

            //double[] result = new double[numOutput];
            //this.outputs.CopyTo(result, 0);
            //return result;

            double[] result = Softmax(_hoSums);

            result.CopyTo(_outputs, 0);

            //Console.WriteLine("outputs after softmaxing:");
            //ShowVector(result);

            //Console.ReadLine();

            //double[] result = Hardmax(hoSums);
            return result;
        } // ComputeOutputs

        //private static double StepFunction(double x)
        //{
        //  if (x > 0.0) return 1.0;
        //  else return 0.0;
        //}

        private static double SigmoidFunction(double x)
        {
            if (x < -45.0) return 0.0;
            if (x > 45.0) return 1.0;
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double[] Softmax(double[] hoSums)
        {
            // determine max
            double max = hoSums.Max();

            // determine scaling factor (sum of exp(eachval - max)
            double scale = hoSums.Sum(t => Math.Exp(t - max));

            var result = hoSums.Select(t => Math.Exp(t - max) / scale);
            return result.ToArray();
        }

        public double[] Train(double[][] trainMatrix) // seek and return the best weights
        {
            int numWeights = (_numInput * _numHidden) + (_numHidden * _numOutput) + _numHidden + _numOutput;
            //double[] currWeights = new double[numWeights];

            // use PSO to seek best weights
            int numberParticles = 10;
            int numberIterations = 500;
            int iteration = 0;
            int dim = numWeights; // number of values to solve for
            double minX = -5.0; // for each weight
            double maxX = 5.0;

            var swarm = new Particle[numberParticles];
            var bestGlobalPosition = new double[dim]; // best solution found by any particle in the swarm. implicit initialization to all 0.0
            double bestGlobalFitness = double.MaxValue; // smaller values better

            double minV = -0.1 * maxX;  // velocities
            double maxV = 0.1 * maxX;

            for (int i = 0; i < swarm.Length; ++i) // initialize each Particle in the swarm with random positions and velocities
            {
                var randomPosition = new double[dim];
                for (int j = 0; j < randomPosition.Length; ++j)
                {
                    double lo = minX;
                    double hi = maxX;
                    randomPosition[j] = (hi - lo) * _rnd.NextDouble() + lo; 
                }

                double fitness = CrossEntropy(trainMatrix, randomPosition); // smaller values better
                var randomVelocity = new double[dim];

                for (int j = 0; j < randomVelocity.Length; ++j)
                {
                    double lo = -1.0 * Math.Abs(maxX - minX);
                    double hi = Math.Abs(maxX - minX);
                    randomVelocity[j] = (hi - lo) * _rnd.NextDouble() + lo;
                }
                swarm[i] = new Particle(randomPosition, fitness, randomVelocity, randomPosition, fitness);

                // does current Particle have global best position/solution?
                if (swarm[i].Fitness < bestGlobalFitness)
                {
                    bestGlobalFitness = swarm[i].Fitness;
                    swarm[i].Position.CopyTo(bestGlobalPosition, 0);
                }
            } // initialization

            double w = 0.729; // inertia weight.
            double c1 = 1.49445; // cognitive/local weight
            double c2 = 1.49445; // social/global weight
            double r1, r2; // cognitive and social randomizations

            Console.WriteLine("Entering main PSO weight estimation processing loop");
            while (iteration < numberIterations)
            {
                ++iteration;
                double[] newVelocity = new double[dim];
                double[] newPosition = new double[dim];
                double newFitness;

                for (int i = 0; i < swarm.Length; ++i) // each Particle
                {
                    Particle currP = swarm[i];

                    for (int j = 0; j < currP.Velocity.Length; ++j) // each x value of the velocity
                    {
                        r1 = _rnd.NextDouble();
                        r2 = _rnd.NextDouble();

                        newVelocity[j] = (w * currP.Velocity[j]) +
                                         (c1 * r1 * (currP.BestPosition[j] - currP.Position[j])) +
                                         (c2 * r2 * (bestGlobalPosition[j] - currP.Position[j])); // new velocity depends on old velocity, best position of parrticle, and best position of any particle

                        if (newVelocity[j] < minV)
                            newVelocity[j] = minV;
                        else if (newVelocity[j] > maxV)
                            newVelocity[j] = maxV;     // crude way to keep velocity in range
                    }

                    newVelocity.CopyTo(currP.Velocity, 0);

                    for (int j = 0; j < currP.Position.Length; ++j)
                    {
                        newPosition[j] = currP.Position[j] + newVelocity[j];  // compute new position
                        if (newPosition[j] < minX)
                            newPosition[j] = minX;
                        else if (newPosition[j] > maxX)
                            newPosition[j] = maxX;
                    }

                    newPosition.CopyTo(currP.Position, 0);

                    newFitness = CrossEntropy(trainMatrix, newPosition);  // compute error of the new position
                    currP.Fitness = newFitness;

                    if (newFitness < currP.BestFitness) // new particle best?
                    {
                        newPosition.CopyTo(currP.BestPosition, 0);
                        currP.BestFitness = newFitness;
                    }

                    if (newFitness < bestGlobalFitness) // new global best?
                    {
                        newPosition.CopyTo(bestGlobalPosition, 0);
                        bestGlobalFitness = newFitness;
                    }

                } // each Particle

                //Console.WriteLine(swarm[0].ToString());
                //Console.ReadLine();

            } // while

            Console.WriteLine("Processing complete");
            Console.Write("Final best (smallest) cross entropy error = ");
            Console.WriteLine(bestGlobalFitness.ToString("F4"));

            return bestGlobalPosition;

        } // Train

        private double CrossEntropy(double[][] trainData, double[] weights) // (sum) Cross Entropy
        {
            // how good (cross entropy) are weights? CrossEntropy is error so smaller values are better
            SetWeights(weights); // load the weights and biases to examine

            double sce = 0.0; // sum of cross entropy

            foreach (double[] t in trainData)
            {
                double[] currInputs = new double[4]; currInputs[0] = t[0]; currInputs[1] = t[1]; currInputs[2] = t[2]; currInputs[3] = t[3];
                double[] currExpected = new double[3]; currExpected[0] = t[4]; currExpected[1] = t[5]; currExpected[2] = t[6]; // not really necessary
        
                double[] currOutputs = ComputeOutputs(currInputs); // run the jnputs through the neural network

                // compute ln of each nn output (and the sum)
                double currSum = 0.0;
                for (int j = 0; j < currOutputs.Length; ++j)
                {
                    if (Math.Abs(currExpected[j] - 0.0) > Epsilon)
                        currSum += currExpected[j] * Math.Log(currOutputs[j]);
                }
                sce += currSum; // accumulate
            }
            return -sce;
        } // CrossEntropy

        public double Test(double[][] testMatrix) // returns the accuracy (percent correct predictions)
        {
            // assumes that weights have been set using SetWeights
            int numCorrect = 0;
            int numWrong = 0;

            for (int i = 0; i < testMatrix.Length; ++i) // walk thru each test case. looks like (6.9 3.2 5.7 2.3) (0 0 1)  where the parens are not really there
            {
        
                double[] currInputs = new double[4]; currInputs[0] = testMatrix[i][0]; currInputs[1] = testMatrix[i][1]; currInputs[2] = testMatrix[i][2]; currInputs[3] = testMatrix[i][3];
                double[] currOutputs = new double[3]; currOutputs[0] = testMatrix[i][4]; currOutputs[1] = testMatrix[i][5]; currOutputs[2] = testMatrix[i][6]; // not really necessary
                double[] currPredicted = ComputeOutputs(currInputs); // outputs are in softmax form -- each between 0.0, 1.0 representing a prob and summing to 1.0

                //ShowVector(currInputs);
                //ShowVector(currOutputs);
                //ShowVector(currPredicted);

                // use winner-takes all -- highest prob of the prediction
                int indexOfLargest = Helpers.IndexOfLargest(currPredicted);

                if (i <= 3) // just a few for demo purposes
                {
                    Console.WriteLine("-----------------------------------");
                    Console.Write("Input:     ");
                    Helpers.ShowVector(currInputs, 2, true);
                    Console.Write("Output:    ");
                    Helpers.ShowVector(currOutputs, 1, false);
                    if (Math.Abs(currOutputs[0] - 1.0) < Epsilon) Console.WriteLine(" (red)");
                    else if (Math.Abs(currOutputs[1] - 1.0) < Epsilon) Console.WriteLine(" (green)");
                    else Console.WriteLine(" (blue)");
                    Console.Write("Predicted: ");
                    Helpers.ShowVector(currPredicted, 1, false);
                    if (indexOfLargest == 0) Console.WriteLine(" (red)");
                    else if (indexOfLargest == 1) Console.WriteLine(" (green)");
                    else Console.WriteLine(" (blue)");

                    if (Math.Abs(currOutputs[indexOfLargest] - 1) < Epsilon)
                        Console.WriteLine("correct");
                    else
                        Console.WriteLine("wrong");
                    Console.WriteLine("-----------------------------------");
                }

                if (Math.Abs(currOutputs[indexOfLargest] - 1) < Epsilon)
                    ++numCorrect;
                else
                    ++numWrong;
         
                //Console.ReadLine();
            }
            Console.WriteLine(". . .");

            double percentCorrect = (numCorrect * 1.0) / (numCorrect + numWrong);
            Console.WriteLine("\nCorrect = " + numCorrect);
            Console.WriteLine("Wrong = " + numWrong);

            return percentCorrect;
        } // Test

    }
}
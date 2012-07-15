using System;
using System.IO;

namespace NeuralClassification
{
    internal class NeuralClassificationProgram
    {
        private static Random _rnd;

        private static void Main()
        {
            try
            {
                Console.WriteLine("\nBegin neural network classification demo\n");
                Console.WriteLine("Goal is to predict/classify color based on four numeric inputs\n");
                _rnd = new Random(159); // 159 makes 'good' output

                Console.WriteLine("Creating 100 lines of raw data");
                string dataFile = "..\\..\\colors.txt";
                MakeData(dataFile, 100);

                Console.WriteLine("\nFirst few rows of raw data file are:");
                Helpers.ShowTextFile(dataFile, 4);

                double[][] trainMatrix;
                double[][] testMatrix;
                Console.WriteLine("\nGenerating train and test matrices using an 80%-20% split");
                MakeTrainAndTest(dataFile, out trainMatrix, out testMatrix);

                Console.WriteLine("\nFirst few rows of training matrix are:");
                Helpers.ShowMatrix(trainMatrix, 5);

                Console.WriteLine("\nCreating 4-input 5-hidden 3-output neural network");
                var nn = new NeuralNetwork(4, 5, 3);

                Console.WriteLine("Training to find best neural network weights using PSO with cross entropy error");
                double[] bestWeights = nn.Train(trainMatrix);
                Console.WriteLine("\nBest weights found:");
                Helpers.ShowVector(bestWeights, 2, true);

                Console.WriteLine("\nLoading best weights into neural network");
                nn.SetWeights(bestWeights);

                Console.WriteLine("\nAnalyzing the neural network accuracy on the test data\n");
                double accuracy = nn.Test(testMatrix);
                Console.WriteLine("Prediction accuracy = " + accuracy.ToString("F4"));

                Console.WriteLine("\nEnd neural network classification demo\n");
                Console.ReadLine();
            }
            catch (Exception ex)
            {
                Console.WriteLine("Fatal: " + ex.Message);
                Console.ReadLine();
            }
        }

        // Main()

        // --------------------------------------------------------------------------------------------

        private static void MakeData(string dataFile, int numLines)
        {
            var weights = new[]
                              {
                                  -0.1, 0.2, -0.3, 0.4, -0.5,
                                  0.6, -0.7, 0.8, -0.9, 1.0,
                                  -1.1, 1.2, -1.3, 1.4, -1.5,
                                  1.6, -1.7, 1.8, -1.9, 2.0,
                                  -0.5, 0.6, -0.7, 0.8, -0.9,
                                  1.5, -1.4, 1.3,
                                  -1.2, 1.1, -1.0,
                                  0.9, -0.8, 0.7,
                                  -0.6, 0.5, -0.4,
                                  0.3, -0.2, 0.1,
                                  0.1, -0.3, 0.6
                              };

            var nn = new NeuralNetwork(4, 5, 3);
            nn.SetWeights(weights);

            var ofs = new FileStream(dataFile, FileMode.Create);
            var sw = new StreamWriter(ofs);

            for (int i = 0; i < numLines; ++i)
            {
                var inputs = new double[4];
                for (int j = 0; j < inputs.Length; ++j)
                    inputs[j] = _rnd.Next(1, 10);

                double[] outputs = nn.ComputeOutputs(inputs);

                string color = "";
                int idx = Helpers.IndexOfLargest(outputs);
                if (idx == 0)
                {
                    color = "red";
                }
                else if (idx == 1)
                {
                    color = "green";
                }
                else if (idx == 2)
                {
                    color = "blue";
                }

                sw.WriteLine(inputs[0].ToString("F1") + " " + inputs[1].ToString("F1") + " " + inputs[2].ToString("F1") +
                             " " + inputs[3].ToString("F1") + " " + color);
            }
            sw.Close();
            ofs.Close();
        }

        // MakeData

        private static void MakeTrainAndTest(string file, out double[][] trainMatrix, out double[][] testMatrix)
        {
            int numLines = 0;
            var ifs = new FileStream(file, FileMode.Open);
            var sr = new StreamReader(ifs);
            while (sr.ReadLine() != null)
                ++numLines;
            sr.Close();
            ifs.Close();

            var numTrain = (int) (0.80*numLines);
            int numTest = numLines - numTrain;

            var allData = new double[numLines][]; // could use Helpers.MakeMatrix here
            for (int i = 0; i < allData.Length; ++i)
                allData[i] = new double[7]; // (x0, x1, x2, x3), (y0, y1, y2)

            string line;
            string[] tokens;
            ifs = new FileStream(file, FileMode.Open);
            sr = new StreamReader(ifs);
            int row = 0;
            while ((line = sr.ReadLine()) != null)
            {
                tokens = line.Split(' ');
                allData[row][0] = double.Parse(tokens[0]);
                allData[row][1] = double.Parse(tokens[1]);
                allData[row][2] = double.Parse(tokens[2]);
                allData[row][3] = double.Parse(tokens[3]);

                for (int i = 0; i < 4; ++i)
                    allData[row][i] = 0.25*allData[row][i] - 1.25; // scale input data to [-1.0, +1.0]

                if (tokens[4] == "red")
                {
                    allData[row][4] = 1.0;
                    allData[row][5] = 0.0;
                    allData[row][6] = 0.0;
                }
                else if (tokens[4] == "green")
                {
                    allData[row][4] = 0.0;
                    allData[row][5] = 1.0;
                    allData[row][6] = 0.0;
                }
                else if (tokens[4] == "blue")
                {
                    allData[row][4] = 0.0;
                    allData[row][5] = 0.0;
                    allData[row][6] = 1.0;
                }
                ++row;
            }
            sr.Close();
            ifs.Close();

            Helpers.ShuffleRows(allData);

            trainMatrix = Helpers.MakeMatrix(numTrain, 7);
            testMatrix = Helpers.MakeMatrix(numTest, 7);

            for (int i = 0; i < numTrain; ++i)
            {
                allData[i].CopyTo(trainMatrix[i], 0);
            }

            for (int i = 0; i < numTest; ++i)
            {
                allData[i + numTrain].CopyTo(testMatrix[i], 0);
            }
        }

        // MakeTrainAndTest

        // --------------------------------------------------------------------------------------------
    }

    // class NeuralClassificationProgram
}

// ns
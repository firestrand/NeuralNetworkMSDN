using System;
using System.IO;

namespace NeuralClassification
{
    public class Helpers
    {
        static readonly Random Rnd = new Random(0);

        public static double[][] MakeMatrix(int rows, int cols)
        {
            var result = new double[rows][];
            for (int i = 0; i < rows; ++i)
                result[i] = new double[cols];
            return result;
        }

        public static void ShuffleRows(double[][] matrix)
        {
            for (int i = 0; i < matrix.Length; ++i)
            {
                int r = Rnd.Next(i, matrix.Length);
                double[] tmp = matrix[r];
                matrix[r] = matrix[i];
                matrix[i] = tmp;
            }
        }

        public static int IndexOfLargest(double[] vector)
        {
            int indexOfLargest = 0;
            double maxVal = vector[0];
            for (int i = 0; i < vector.Length; ++i)
            {
                if (vector[i] > maxVal)
                {
                    maxVal = vector[i];
                    indexOfLargest = i;
                }
            }
            return indexOfLargest;
        }

        public static void ShowVector(double[] vector, int decimals, bool newLine)
        {
            string fmt = "F" + decimals;
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i > 0 && i % 12 == 0)
                    Console.WriteLine("");
                if (vector[i] >= 0.0) Console.Write(" ");
                Console.Write(vector[i].ToString(fmt) + " ");
            }
            if (newLine) Console.WriteLine("");
        }

        public static void ShowMatrix(double[][] matrix, int numRows)
        {
            int ct = 0;
            if (numRows == -1) numRows = int.MaxValue;
            for (int i = 0; i < matrix.Length && ct < numRows; ++i)
            {
                for (int j = 0; j < matrix[0].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0) Console.Write(" ");
                    if (j == 4) Console.Write("-> ");
                    Console.Write(matrix[i][j].ToString("F2") + " ");
                }
                Console.WriteLine("");
                ++ct;
            }
            Console.WriteLine("");
        }

        public static void ShowTextFile(string textFile, int numLines)
        {
            var ifs = new FileStream(textFile, FileMode.Open);
            var sr = new StreamReader(ifs);
            string line;
            int ct = 0;
            while ((line = sr.ReadLine()) != null && ct < numLines)
            {
                Console.WriteLine(line);
                ++ct;
            }
            sr.Close(); ifs.Close();
        }

    }
}
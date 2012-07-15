namespace NeuralClassification
{
    public class Particle
    {
        public double BestFitness;
        public double[] BestPosition; // best position found so far by this Particle
        public double Fitness;
        public double[] Position; // equivalent to x-Values and/or solution
        public double[] Velocity;

        public Particle(double[] position, double fitness, double[] velocity, double[] bestPosition, double bestFitness)
        {
            Position = new double[position.Length];
            position.CopyTo(Position, 0);
            Fitness = fitness;
            Velocity = new double[velocity.Length];
            velocity.CopyTo(Velocity, 0);
            BestPosition = new double[bestPosition.Length];
            bestPosition.CopyTo(BestPosition, 0);
            BestFitness = bestFitness;
        }

        public override string ToString()
        {
            string s = "";
            s += "==========================\n";
            s += "Position: ";
            for (int i = 0; i < Position.Length; ++i)
                s += Position[i].ToString("F2") + " ";
            s += "\n";
            s += "Fitness = " + Fitness.ToString("F4") + "\n";
            s += "Velocity: ";
            for (int i = 0; i < Velocity.Length; ++i)
                s += Velocity[i].ToString("F2") + " ";
            s += "\n";
            s += "Best Position: ";
            for (int i = 0; i < BestPosition.Length; ++i)
                s += BestPosition[i].ToString("F2") + " ";
            s += "\n";
            s += "Best Fitness = " + BestFitness.ToString("F4") + "\n";
            s += "==========================\n";
            return s;
        }
    }
}
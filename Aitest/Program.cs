using System;
using System.Text;
using System.Collections.Generic;
using System.Linq;

namespace AI
{
    /// <summary>
    /// 
    /// </summary>
    class Program
    {
        static void Main(string[] args)
        {
            List<double[]> lstInput = new List<double[]>();
            lstInput.Add(new double[] { 0.0341754, 0.0359450, 0.0320310, 0.0360899, 0.0339913 });
            lstInput.Add(new double[] { 0.0338697, 0.0327013, 0.0321764, 0.0321457, 0.0319385 });
            lstInput.Add(new double[] { 0.0332281, 0.0320809, 0.0310662, 0.0315236, 0.0323512 });
            lstInput.Add(new double[] { 0.0348996, 0.0363309, 0.0378850, 0.0383740, 0.0366937 });
            lstInput.Add(new double[] { 0.0718842, 0.0800624, 0.0763155, 0.0722577, 0.0713689 });
            lstInput.Add(new double[] { 0.0747161, 0.0731235, 0.0752407, 0.0845412, 0.0850135 });
            lstInput.Add(new double[] { 0.0943182, 0.0959164, 0.0920233, 0.1002110, 0.1921506 });
            lstInput.Add(new double[] { -0.0858807, 0.1038390, 0.2021741, 0.1127879, 0.1435561 });

            List<double[]> lstOutput = new List<double[]>();
            lstOutput.Add(new double[] { 0.0332516 });
            lstOutput.Add(new double[] { 0.0331811 });
            lstOutput.Add(new double[] { 0.0337170 });
            lstOutput.Add(new double[] { 0.0353268 });
            lstOutput.Add(new double[] { 0.0714258 });
            lstOutput.Add(new double[] { 0.0913901 });
            lstOutput.Add(new double[] { 0.0963110 });
            lstOutput.Add(new double[] { 0.1163857 });

            FeedForwardNeuralNetwork myNN = new FeedForwardNeuralNetwork();
            myNN.Init(5, 3, 1);
            Tuple<List<double[]>, List<double[]>> pat = Tuple.Create(lstInput, lstOutput);
            myNN.Train(pat);

            List<double[]> lstTestInput = new List<double[]>();
            lstTestInput.Add(new double[] { 0.1369399, 0.0924755, 0.0916544, 0.0926251, 0.0921044 });
            List<double[]> lstTestOutput = new List<double[]>();
            lstTestOutput.Add(new double[] { 0.0933581 });
            Tuple<List<double[]>, List<double[]>> patTest = Tuple.Create(lstTestInput, lstTestOutput);
            myNN.Test(patTest);

            Console.ReadLine();
        }
    }
}
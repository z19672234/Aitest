using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    /// <summary>
    /// 神经网络单元
    /// </summary>
    public class NeuralNetworkUnit
    {
        /// <summary>
        /// 获取神经网络单元输出值
        /// </summary>
        /// <param name="x">输入变量 x</param>
        /// <param name="w">权重 w（weight）</param>
        /// <param name="b">偏置</param>
        /// <returns>返回结果y</returns>
        public static double GetSigmoidValue(double[] x, double[] w, double b)
        {
            return ActivationFunction.Sigmoid(SigmaWeight(x, w) + b);
        }

        /// <summary>
        /// 求和 Σ
        /// </summary>
        /// <param name="x">输入变量 x</param>
        /// <param name="w">权重 w（weight）</param>
        /// <returns>返回求和结果</returns>
        public static double SigmaWeight(double[] x, double[] w)
        {
            double total = 0;
            for (int i = 0; i < x.Length; i++)
            {
                total += x[i] * w[i];
            }
            return total;
        }
    }
}

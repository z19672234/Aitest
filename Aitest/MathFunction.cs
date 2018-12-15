using System;

namespace AI
{
    /// <summary>
    /// 常用数学函数
    /// </summary>
    public class MathFunction
    {
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

        /// <summary>
        /// 对Sigmoid函数求导
        /// Delta Δ
        /// </summary>
        /// <param name="x">输入值</param>
        /// <returns>输出求导后的结果</returns>
        public static double DerivativeSigmoid(double x)
        {
            return x * (1 - x);
        }

        /// <summary>
        /// 创建矩阵
        /// </summary>
        /// <param name="r">行数量</param>
        /// <param name="c">列数量</param>
        /// <param name="fill">默认填充值</param>
        /// <returns>返回二维矩阵</returns>
        public static double[][] MakeMatrix(int r, int c, double fill = 0.0)
        {
            double[][] m = new double[r][];
            for (int i = 0; i < r; i++)
            {
                double[] array = new double[c];
                for (int j = 0; j < c; j++)
                {
                    array[j] = fill;
                }
                m[i] = array;
            }
            return m;
        }

        /// <summary>
        /// 初始化矩阵
        /// </summary>
        /// <param name="matrix">矩阵</param>
        /// <param name="min">随机最小值</param>
        /// <param name="max">随机最大值</param>
        /// <returns>返回矩阵</returns>
        public static double[][] RandomizeMatrix(double[][] matrix, double min = 0, double max = 1)
        {
            for (int i = 0; i < matrix.Length; i++)
            {
                for (int j = 0; j < matrix[0].Length; j++)
                {
                    Random r = new Random();
                    matrix[i][j] = r.NextDouble() * (max - min) + min;
                }
            }

            return matrix;
        }
    }
}

using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    /// <summary>
    /// 前馈神经网络(FeedForward Neural Network)
    /// 由于它一般是要向后传递误差的，所以也叫误差反向传播神经网络(Error Back Propagation Neural Network)，简称BP神经网络
    /// 
    /// 算法参考：https://github.com/hankcs/neural_net
    /// </summary>
    public class FeedForwardNeuralNetwork
    {
        /// <summary>
        /// 输入单元
        /// </summary>
        private double[] _ai;

        /// <summary>
        /// 隐藏单元
        /// </summary>
        private double[] _ah;

        /// <summary>
        /// 输出单元
        /// </summary>
        private double[] _ao;

        /// <summary>
        /// 输入单元数量
        /// </summary>
        private int _ni;

        /// <summary>
        /// 隐藏单元数量
        /// </summary>
        private int _nh;

        /// <summary>
        /// 输出单元数量
        /// </summary>
        private int _no;

        /// <summary>
        /// 输入层到隐藏层权重矩阵
        /// </summary>
        private double[][] _wi;

        /// <summary>
        /// 隐藏层到输出层权重矩阵
        /// </summary>
        private double[][] _wo;

        /// <summary>
        /// 权重矩阵的上次梯度-输入层到隐藏层
        /// </summary>
        private double[][] _ci;

        /// <summary>
        /// 权重矩阵的上次梯度-隐藏层到输出层
        /// </summary>
        private double[][] _co;

        /// <summary>
        /// 构造神经网络
        /// </summary>
        /// <param name="ni">输入单元数量</param>
        /// <param name="nh">隐藏单元数量</param>
        /// <param name="no">输出单元数量</param>
        public void Init(int ni, int nh, int no)
        {
            //各层单元数量
            this._ni = ni + 1;
            this._nh = nh;
            this._no = no;

            //激活值（输出值）
            this._ai = new double[this._ni];
            this._ah = new double[this._nh];
            this._ao = new double[this._no];

            //创建权重矩阵
            this._wi = MathFunction.MakeMatrix(this._ni, this._nh);
            this._wo = MathFunction.MakeMatrix(this._nh, this._no);

            //将权重矩阵随机化
            MathFunction.RandomizeMatrix(this._wi);
            MathFunction.RandomizeMatrix(this._wo, -1, 1);

            //权重矩阵的上次梯度
            this._ci = MathFunction.MakeMatrix(this._ni, this._nh);
            this._co = MathFunction.MakeMatrix(this._nh, this._no);
        }

        /// <summary>
        /// 前向传播进行分类
        /// </summary>
        /// <param name="inputs">输入</param>
        /// <returns>类别</returns>
        public double[] RunNN(double[] inputs)
        {
            if (inputs.Length != (this._ni - 1))
                Console.WriteLine("incorrect number of inputs");

            for (int i = 0; i < (this._ni - 1); i++)
            {
                this._ai[i] = inputs[i];
            }

            for (int j = 0; j < this._nh; j++)
            {
                double sum = 0.0;
                for (int i = 0; i < this._ni; i++)
                {
                    sum += this._ai[i] * this._wi[i][j];
                }
                this._ah[j] = ActivationFunction.Sigmoid(sum);
            }

            for (int k = 0; k < this._no; k++)
            {
                double sum = 0.0;
                for (int j = 0; j < this._nh; j++)
                {
                    sum += this._ah[j] * this._wo[j][k];
                }
                this._ao[k] = ActivationFunction.Sigmoid(sum);
            }

            return this._ao;
        }

        /// <summary>
        /// 后向传播算法
        /// </summary>
        /// <param name="targets"></param>
        /// <param name="N"></param>
        /// <param name="M"></param>
        /// <returns></returns>
        public double BackPropagate(double[] targets, double N, double M)
        {
            //计算输出层 deltas
            double[] output_deltas = new double[this._no];
            for (int k = 0; k < this._no; k++)
            {
                double oerror = targets[k] - this._ao[k];
                output_deltas[k] = oerror * MathFunction.DerivativeSigmoid(this._ao[k]);
            }

            //更新输出层权值
            for (int j = 0; j < this._nh; j++)
            {
                for (int k = 0; k < this._no; k++)
                {
                    double change = output_deltas[k] * this._ah[j];
                    this._wo[j][k] += N * change + M * this._co[j][k];
                    this._co[j][k] = change;
                }
            }

            //计算隐藏层 deltas
            double[] hidden_deltas = new double[this._nh];
            for (int j = 0; j < this._nh; j++)
            {
                double herror = 0.0;
                for (int k = 0; k < this._no; k++)
                {
                    herror += output_deltas[k] * this._wo[j][k];
                }
                hidden_deltas[j] = herror * MathFunction.DerivativeSigmoid(this._ah[j]);
            }

            //更新输入层权值
            for (int i = 0; i < this._ni; i++)
            {
                for (int j = 0; j < this._nh; j++)
                {
                    double change = hidden_deltas[j] * this._ai[i];
                    this._wi[i][j] += N * change + M * this._ci[i][j];
                    this._ci[i][j] = change;
                }
            }

            //计算误差平方和
            double error = 0.0;
            for (int k = 0; k < targets.Length; k++)
            {
                error = 0.5 * Math.Pow((targets[k] - this._ao[k]), 2);
            }
            return error;
        }

        /// <summary>
        /// 打印权值矩阵
        /// </summary>
        public void Weights()
        {
            for (int i = 0; i < this._ni; i++)
            {
                Console.WriteLine(this._wi[i]);
            }
            Console.WriteLine();

            for (int j = 0; j < this._nh; j++)
            {
                Console.WriteLine(this._wo[j]);
            }
            Console.WriteLine();
        }

        /// <summary>
        /// 测试
        /// </summary>
        /// <param name="patterns"></param>
        public void Test(Tuple<List<double[]>, List<double[]>> patterns)
        {
            for(int p = 0; p < patterns.Item1.Count; p++)
            {
                double[] inputs = patterns.Item1[p];

                StringBuilder sbInput = new StringBuilder();
                sbInput.Append("[");
                foreach (var input in inputs)
                {
                    sbInput.Append(input).Append(",");
                }
                string strInput = sbInput.ToString().TrimEnd(',') + "]";

                StringBuilder sbOutput = new StringBuilder();
                sbOutput.Append("[");
                foreach (var output in patterns.Item2[p])
                {
                    sbOutput.Append(output).Append(",");
                }
                string strOutput = sbOutput.ToString().TrimEnd(',') + "]";

                double[] results = this.RunNN(inputs);
                StringBuilder sbResult = new StringBuilder();
                foreach(var r in results)
                {
                    sbResult.Append(r).Append(","); ;
                }
                string strResult = sbResult.ToString().TrimEnd(',');

                Console.WriteLine("Inputs:" + strInput +  "--->"+ strResult + "\tTarget" + strOutput);
            }
        }

        /// <summary>
        /// 训练
        /// </summary>
        /// <param name="patterns">训练集</param>
        /// <param name="max_iterations">最大迭代次数</param>
        /// <param name="N">本次学习率</param>
        /// <param name="M">上次学习率</param>
        public void Train(Tuple<List<double[]>, List<double[]>> patterns, int max_iterations = 10000, double N = 0.25, double M = 0.12)
        {
            for (int i = 0; i < max_iterations; i++)
            {
                for (int p=0; p<patterns.Item1.Count; p++)
                {
                    double[] inputs = patterns.Item1[p];
                    double[] targets = patterns.Item2[p];
                    RunNN(inputs);
                    double error = BackPropagate(targets, N, M);
                    Console.WriteLine("iterations: " +i + " error:" + error.ToString("f10"));
                }
            }
        }
    }
}

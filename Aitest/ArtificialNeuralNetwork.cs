using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    /// <summary>
    /// 人工神经网络(Artificial Neural Network)
    /// 
    /// Demo
    /// </summary>
    public class ArtificialNeuralNetwork
    {
        /// <summary>
        /// 输入变量x
        /// </summary>
        private double[] _x;

        /// <summary>
        /// 输出变量y
        /// </summary>
        private double[] _y;

        /// <summary>
        /// 偏置
        /// </summary>
        private double _bias = 0.1;

        /// <summary>
        /// 神经网络构造函数
        /// </summary>
        /// <param name="inputX">输入x值</param>
        /// <param name="outputY">输出y值</param>
        /// <param name="bias">偏置</param>
        public ArtificialNeuralNetwork(double[] inputX, double[] outputY)
        {
            _x = inputX;
            _y = outputY;
        }

        /// <summary>
        /// 训练模型
        /// </summary>
        public void Train()
        {
            //获取隐藏层节点个数
            int hdRootNum = HiddenLayer.GetNumByLog(_x.Length);

            //定义存放隐藏层节点单元数组
            double[] hdRootValues = new double[hdRootNum];

            double[][] xh = new double[_x.Length][];

            //对隐藏层节点进行循环
            for (int i = 0; i < hdRootNum; i++)
            {
                //1、初始化每一个输入层x到隐藏层hd节点的权重
                xh[i] = Weight.Initialize(_x.Length);

                for (int j = 0; j < xh[i].Length; j++)
                {
                    Console.WriteLine("The values of x2h weight  is " + xh[i][j]);
                }
                //2、对输入层 ->隐藏层节点进行加权求和  3、求出隐藏层的sigmoid激活函数值
                hdRootValues[i] = NeuralNetworkUnit.GetSigmoidValue(_x, xh[i], _bias);
                Console.WriteLine("The values of hidden layer  is " + hdRootValues[i]);
            }

            //求出输出节点值
            double[] yRootValues = new double[_y.Length];

            double[][] hy = new double[_y.Length][];

            for (int i = 0; i < _y.Length; i++)
            {
                //初始化每一个隐藏层hd节点到输出层y的权重
                hy[i] = Weight.Initialize(hdRootValues.Length);

                for (int j = 0; j < hy[i].Length; j++)
                {
                    Console.WriteLine("The values of h2y weight  is " + hy[i][j]);
                }
                //4、输出层节点进行加权求和  5、求出输出层的sigmoid激活函数值
                yRootValues[i] = NeuralNetworkUnit.GetSigmoidValue(hdRootValues, hy[i], _bias);
                Console.WriteLine("The values of output layer is " + yRootValues[i]);
            }
        }
    }
}

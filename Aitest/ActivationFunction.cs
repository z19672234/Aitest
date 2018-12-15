using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    /// <summary>
    /// 激活函数（Activation Function）
    /// 
    /// 人工神经网络的神经元上运行的函数，负责将神经元的输入映射到输出端，常见的激活函数
    /// </summary>
    public class ActivationFunction
    {
        /// <summary>
        /// 阀值型函数
        /// 
        /// 公式：f(x)={1 x>=0; 0 x 小于 0}
        /// </summary>
        /// <param name="x">输入变量x</param>
        /// <returns>返回函数值</returns>
        public static int Threshold(double x)
        {
            return x >= 0 ? 1 : 0;
        }

        /// <summary>
        /// 线性函数
        /// 
        /// 公式：f(x) = a * x + b
        /// </summary>
        /// <param name="x">输入变量x</param>
        /// <param name="a">系数</param>
        /// <param name="b">偏置</param>
        /// <returns>返回函数值</returns>
        public static double Linear(double x, double a, double b)
        {
            return x * a + b;
        }

        /// <summary>
        /// 线性 ReLU 函数
        /// 
        /// 优点：（1）因为是线性，而且梯度不会饱和，所以收敛速度会比Sigmoid/tanh快很多；
        ///           （2）相比于Sigmoid/tanh需要计算指数等，计算复杂度高，ReLU只需要一个阈值就可以得到激活值；
        /// 缺点：训练的时候很脆弱，有可能导致神经元坏死。
        ///           举个例子：由于ReLU在x小于0时梯度为0，这样就导致负的梯度在这个ReLU被置零，而且这个神经元有可能再也不会被任何数据激活。
        ///           如果这个情况发生了，那么这个神经元之后的梯度就永远是0了，也就是ReLU神经元坏死了，不再对任何数据有所响应。
        ///           实际操作中，如果你的learning rate 很大，那么很有可能你网络中的40%的神经元都坏死了。
        /// </summary>
        /// <param name="x">输入变量x</param>
        /// <returns>返回函数值</returns>
        public static double ReLU(double x)
        {
            return x > 0 ? x : 0;
        }

        /// <summary>
        /// 非线性-激活函数 Sigmoid  单极性S型函数 
        /// 
        /// 公式：f(x)=1/1+exp(-x)
        /// 
        /// 优点：能够把输入的连续实值压缩到0到1之间；
        /// 缺点：（1）容易饱和，当输入非常大或非常小的时候，神经元的梯度就接近0了，这使得在反向传播算法中反向传播接近于0的梯度，导致最终权重基本没什么更新；
        ///           （2）Sigmoid的输出不是0均值的，这会导致后层的神经元的输入是非0均值的信号，这会对梯度产生影响，
        ///                     假设后层神经元的输入都为正(e.g.x>0elementwise in f= wTx + b),那么对w求局部梯度则都为正，这样在反向传播的过程中w要么都往正方向更新，
        ///                     要么都往负方向更新，导致有一种捆绑的效果，使得收敛缓慢。   
        /// 解决方法：注意参数的初始值设置来避免饱和情况。
        /// </summary>
        /// <param name="x">输入变量x</param>
        /// <returns>返回函数值</returns>
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        /// <summary>
        /// 非线性-激活函数 Tanh 双曲正切S型函数
        /// 
        /// 公式：f(x)=exp(x) - exp(-x)/exp(x)+exp(-x)
        /// 
        /// 
        /// 优点：0均值，能够压缩数据到-1到1之间；
        /// 缺点：同Sigmoid缺点第二个，梯度饱和；
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public static double Tanh(double x)
        {
            return Math.Tanh(x);
            //return 2 * Sigmoid(2 * x) - 1;
        }
    }
}

using System;

namespace AI
{
    /// <summary>
    /// 隐藏层
    /// 
    /// m：隐藏层数量
    /// n：输入节点数
    /// l：输出节点数
    /// α：1-10之间的常数（阿尔法 α）
    /// </summary>
    public class HiddenLayer
    {
        /// <summary>
        /// 输入节点数乘以输出节点数的平方根
        /// 
        /// 公式：Sqrt(n * l)
        /// </summary>
        /// <param name="n">输入节点数</param>
        /// <param name="l">输出节点数</param>
        /// <returns></returns>
        public static int GetNumBySqrt(double n, double l)
        {
            double d = Math.Sqrt(n * l);
            return Convert.ToInt16(d);
        }

        /// <summary>
        /// 输入节点数加上输出节点数的平方根，再加上1-10之间的常数
        /// 
        /// 公式：Sqrt(n+l) + a
        /// </summary>
        /// <param name="n">输入节点数</param>
        /// <param name="l">输出节点数</param>
        /// <param name="α">阿尔法，1-10之间的常数</param>
        /// <returns></returns>
        public static int GetNumBySqrt(double n, double l, double α)
        {
            double d = Math.Sqrt(n + l) + α;
            return Convert.ToInt16(d);
        }

        /// <summary>
        /// 输入节点数的对数
        /// 
        /// 公式：Log(n, 2)
        /// </summary>
        /// <param name="n">输入节点数</param>
        /// <returns></returns>
        public static int GetNumByLog(double n)
        {
            double d = Math.Log(n, 2);
            return Convert.ToInt16(d);
        }
    }
}

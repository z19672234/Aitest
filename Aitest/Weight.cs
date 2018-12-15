using System;
using System.Collections.Generic;
using System.Text;

namespace AI
{
    /// <summary>
    /// 权重
    /// </summary>
    public class Weight
    {
        /// <summary>
        /// 初始化权重
        /// 
        /// 返回一个大于或等于 0.0 且小于 1.0 的随机浮点数组
        /// </summary>
        /// <param name="len">输入x的长度</param>
        /// <returns> 返回介于0.0-1.0 之间随机浮点数组</returns>
        public static double[] Initialize(int len)
        {
            double[] d = new double[len];
            for (int i = 0; i < len; i++)
            {
                Random rd = new Random();
                d[i] = rd.NextDouble();
            }
            return d;
        }
    }
}

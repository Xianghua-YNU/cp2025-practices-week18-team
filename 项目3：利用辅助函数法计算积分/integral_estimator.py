import numpy as np
import math

# 1. 推导并实现随机数变换公式：从均匀分布 U(0,1) 生成 p(x) 分布的随机数
def transform_uniform_to_p(xi):
    """
    将均匀分布随机数 xi（~U(0,1)）变换为满足 p(x)=1/(2√x) 分布的随机数 x
    推导：
    p(x) 的累积分布函数 (CDF) 为 F(x) = ∫(0到x) p(t)dt = ∫(0到x) 1/(2√t) dt = √x 
    令 F(x) = xi（均匀分布随机数），则 √x = xi → x = xi²
    """
    return xi ** 2

# 2. 蒙特卡洛积分估计
def monte_carlo_integral(N):
    # 生成 N 个 [0,1] 均匀分布随机数
    xi_samples = np.random.uniform(0, 1, N)  
    # 变换得到 p(x) 分布的随机数
    x_samples = transform_uniform_to_p(xi_samples)  
    
    # 定义被积函数 f(x) = x^(-1/2)/(e^x + 1)
    def f(x):
        return x**(-1/2) / (np.exp(x) + 1)
    
    # 计算权重 w = f(x)/p(x)，这里 p(x)=1/(2√x)，所以 w = 2√x * f(x) = 2 / (e^x + 1)
    w = 2 / (np.exp(x_samples) + 1)  
    
    # 蒙特卡洛积分估计：I ≈ (1/N) * Σw
    integral_estimate = np.mean(w)  
    return integral_estimate

# 3. 计算统计误差
def compute_variance(N, integral_estimate):
    # 重新生成样本计算方差（也可用单次大样本近似）
    xi_samples = np.random.uniform(0, 1, N)
    x_samples = transform_uniform_to_p(xi_samples)
    w = 2 / (np.exp(x_samples) + 1)
    
    # 计算样本方差：var_f = <f²> - <f>²
    f_sq_mean = np.mean(w**2)
    f_mean_sq = integral_estimate ** 2
    var_f = f_sq_mean - f_mean_sq  
    
    # 统计误差 σ = √(var_f / N)
    sigma = math.sqrt(var_f / N)  
    return sigma

# 主流程
if __name__ == "__main__":
    N = 1000000  # 采样数
    # 蒙特卡洛积分估计
    result = monte_carlo_integral(N)  
    # 计算统计误差
    error = compute_variance(N, result)  
    
    print(f"积分估计值: I ≈ {result:.6f}")
    print(f"统计误差: σ ≈ {error:.6f}")

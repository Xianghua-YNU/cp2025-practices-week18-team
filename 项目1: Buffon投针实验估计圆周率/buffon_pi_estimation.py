
import numpy as np
import matplotlib.pyplot as plt


def buffon_experiment(n_trials, l=1.0, d=2.0):
    """
    模拟 Buffon 投针实验

    参数:
    n_trials (int): 投针次数
    l (float): 针的长度
    d (float): 平行线之间的距离 (d >= l)

    返回:
    float: π 的估计值
    int: 相交次数
    """
    # 生成随机数
    x = np.random.uniform(0, d / 2, n_trials)  # 针的中心到最近线的距离
    theta = np.random.uniform(0, np.pi / 2, n_trials)  # 针与线的夹角

    # 判断相交条件
    intersections = x <= (l / 2) * np.sin(theta)
    n_hits = np.sum(intersections)

    # 计算 π 的估计值
    if n_hits == 0:
        return None, 0  # 避免除以零的情况
    pi_estimate = (2 * l * n_trials) / (n_hits * d)

    return pi_estimate, n_hits


# 示例运行
# 设置不同的实验次数
n_trials_list = [100, 1000, 10000, 100000, 1000000]
n_experiments = len(n_trials_list)

# 存储结果
pi_estimates = []
errors = []
n_hits_list = []

# 运行多次实验
for n_trials in n_trials_list:
    pi_estimate, n_hits = buffon_experiment(n_trials)
    pi_estimates.append(pi_estimate)
    n_hits_list.append(n_hits)
    errors.append(abs(pi_estimate - np.pi))

    print(f"实验次数: {n_trials}")
    print(f"  相交次数: {n_hits}")
    print(f"  π 估计值: {pi_estimate:.6f}")
    print(f"  绝对误差: {errors[-1]:.6f}")
    print()

# 绘制结果图表
plt.figure(figsize=(12, 5))

# 左图：π 估计值 vs 实验次数
plt.subplot(1, 2, 1)
plt.plot(n_trials_list, pi_estimates, 'o-', label='Estimated Value')
plt.axhline(y=np.pi, color='r', linestyle='--', label='True Value of π')
plt.xscale('log')
plt.xlabel('Number of Trials (Log Scale)')
plt.ylabel('Estimated Value of π')
plt.title('Estimation of π with Different Number of Trials')
plt.legend()
plt.grid(True)

# 右图：绝对误差 vs 实验次数
plt.subplot(1, 2, 2)
plt.plot(n_trials_list, errors, 'o-', color='g')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of Trials (Log Scale)')
plt.ylabel('Absolute Error (Log Scale)')
plt.title('Estimation Error with Different Number of Trials')
plt.grid(True)

plt.tight_layout()
plt.show()

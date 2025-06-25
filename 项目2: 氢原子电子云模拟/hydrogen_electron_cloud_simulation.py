import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 物理常数 (单位: nm)
a = 5.29e-2  # 玻尔半径 (nm)
r0 = 0.25     # 收敛半径 (nm)
r_max = 1.5 * r0  # 采样最大半径

# 计算正确的空间概率密度最大值
ρ_max = 1/(np.pi * a**3)  # 在r=0处
print(f"正确的空间概率密度最大值: {ρ_max:.2e} nm⁻³")

# 空间概率密度函数 (单位体积的概率)
def probability_density(r):
    return (1/(np.pi * a**3)) * np.exp(-2 * r / a)

# 径向概率密度函数 (用于验证)
def radial_density(r):
    return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# 高效蒙特卡洛采样
def efficient_sampling(n_samples):
    samples = []
    total_trials = 0
    
    # 预计算径向分布的CDF
    r_vals = np.linspace(0, r_max, 1000)
    dr = r_vals[1] - r_vals[0]
    pdf_r = 4 * np.pi * r_vals**2 * probability_density(r_vals)
    cdf = np.cumsum(pdf_r) * dr
    cdf /= cdf[-1]  # 归一化
    
    # 逆变换采样函数
    def sample_r(n):
        u = np.random.rand(n)
        return np.interp(u, cdf, r_vals)
    
    # 批量采样
    batch_size = min(10000, n_samples)
    while len(samples) < n_samples:
        total_trials += batch_size
        
        # 采样径向距离
        r_samples = sample_r(batch_size)
        
        # 采样角度 (球面均匀分布)
        cos_theta = np.random.uniform(-1, 1, batch_size)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2*np.pi, batch_size)
        
        # 转换为直角坐标
        x = r_samples * sin_theta * np.cos(phi)
        y = r_samples * sin_theta * np.sin(phi)
        z = r_samples * cos_theta
        
        samples.extend(np.column_stack((x, y, z)))
        
        # 调整最后一批次大小
        if len(samples) > n_samples:
            samples = samples[:n_samples]
            break
    
    acceptance_rate = n_samples / total_trials
    print(f"采样效率: {acceptance_rate:.2%} (总尝试次数: {total_trials})")
    return np.array(samples)

# 生成电子位置样本
n_samples = 20000
print("开始采样...")
start_time = time.time()
electron_positions = efficient_sampling(n_samples)
print(f"采样完成! 耗时: {time.time()-start_time:.2f}秒")

# 可视化电子云
def plot_electron_cloud(positions):
    fig = plt.figure(figsize=(15, 10))
    
    # 3D 电子云
    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               s=1, alpha=0.1, c='blue')
    ax1.scatter([0], [0], [0], s=100, c='red', marker='o', label='Nucleus')
    ax1.set_xlim([-r_max, r_max])
    ax1.set_ylim([-r_max, r_max])
    ax1.set_zlim([-r_max, r_max])
    ax1.set_xlabel('X (nm)')
    ax1.set_ylabel('Y (nm)')
    ax1.set_zlabel('Z (nm)')
    ax1.set_title(f'Hydrogen Atom Electron Cloud (n={n_samples} samples)')
    ax1.legend()
    
    # 径向分布
    ax2 = fig.add_subplot(222)
    r = np.linalg.norm(positions, axis=1)
    hist, bins = np.histogram(r, bins=50, range=(0, r_max), density=True)
    bin_centers = (bins[:-1] + bins[1:])/2
    
    # 理论曲线
    r_vals = np.linspace(0, r_max, 200)
    theoretical = radial_density(r_vals)
    
    # 归一化理论曲线
    integral = np.trapz(theoretical, r_vals)
    theoretical /= integral
    
    ax2.plot(bin_centers, hist, 'bo', markersize=4, label='Simulation')
    ax2.plot(r_vals, theoretical, 'r-', linewidth=2, label='Theory')
    ax2.set_xlabel('Radius (nm)')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('Radial Probability Distribution')
    ax2.legend()
    ax2.grid(True)
    
    # 2D XY投影
    ax3 = fig.add_subplot(223)
    ax3.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.1, c='blue')
    ax3.set_xlabel('X (nm)')
    ax3.set_ylabel('Y (nm)')
    ax3.set_title('XY Plane Projection')
    ax3.axis('equal')
    ax3.grid(True)
    
    # 角度分布
    ax4 = fig.add_subplot(224, projection='polar')
    phi = np.arctan2(positions[:, 1], positions[:, 0])
    ax4.hist(phi, bins=100, density=True, alpha=0.7)
    ax4.set_title('Azimuthal Distribution (Should be Uniform)')
    
    plt.tight_layout()
    plt.savefig('hydrogen_electron_cloud.png', dpi=150)
    plt.show()

# 执行可视化
plot_electron_cloud(electron_positions)

# 分析不同参数的影响
def analyze_parameters():
    # 测试不同样本数量
    sample_sizes = [1000, 5000, 10000, 20000]
    
    plt.figure(figsize=(14, 10))
    
    for i, n in enumerate(sample_sizes, 1):
        print(f"\n采样数量: {n}")
        positions = efficient_sampling(n)
        r = np.linalg.norm(positions, axis=1)
        
        plt.subplot(2, 2, i)
        hist, bins, _ = plt.hist(r, bins=50, range=(0, r_max), density=True, alpha=0.7)
        bin_centers = (bins[:-1] + bins[1:])/2
        
        # 理论曲线
        r_vals = np.linspace(0, r_max, 200)
        theoretical = radial_density(r_vals)
        theoretical /= np.trapz(theoretical, r_vals)  # 归一化
        
        plt.plot(r_vals, theoretical, 'r-', linewidth=2)
        plt.title(f'Samples = {n}')
        plt.xlabel('Radius (nm)')
        plt.ylabel('Probability Density')
        plt.grid(True)
        
        # 计算并显示误差
        f = np.interp(bin_centers, r_vals, theoretical)
        valid = f > 1e-5
        if np.any(valid):
            error = np.mean(np.abs(hist[valid] - f[valid]) / f[valid])
            plt.annotate(f'Avg Error: {error:.2%}', xy=(0.6, 0.9), 
                         xycoords='axes fraction', fontsize=10)
    
    plt.suptitle('Effect of Sample Size on Radial Distribution', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('sample_size_effect.png', dpi=150)
    plt.show()

# 执行参数分析
analyze_parameters()

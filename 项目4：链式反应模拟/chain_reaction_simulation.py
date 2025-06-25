import numpy as np
import matplotlib.pyplot as plt

# 参数设定
A0 = 1.0       # 初始A浓度
B0 = 0.0       # 初始B浓度
C0 = 0.0       # 初始C浓度
k1 = 0.1       # A -> B的反应速率常数
k2 = 0.05      # B -> C的反应速率常数

# 时间参数
t_max = 100     # 最大模拟时间
dt = 0.1        # 时间步长
n_steps = int(t_max / dt)

# 反应模拟
def chain_reaction(A0, B0, C0, k1, k2, dt, n_steps):
    A, B, C = A0, B0, C0
    A_list, B_list, C_list, time_list = [], [], [], []
    
    for step in range(n_steps):
        # 计算反应速率
        r1 = k1 * A          # A -> B反应速率
        r2 = k2 * B          # B -> C反应速率
        
        # 更新浓度
        A -= r1 * dt
        B += (r1 - r2) * dt
        C += r2 * dt
        
        # 保存数据
        A_list.append(A)
        B_list.append(B)
        C_list.append(C)
        time_list.append(step * dt)
    
    return time_list, A_list, B_list, C_list

# 模拟链式反应
time, A_conc, B_conc, C_conc = chain_reaction(A0, B0, C0, k1, k2, dt, n_steps)

# 绘制结果
plt.figure(figsize=(10,6))
plt.plot(time, A_conc, label='[A]', color='red')
plt.plot(time, B_conc, label='[B]', color='green')
plt.plot(time, C_conc, label='[C]', color='blue')
plt.title('Chain Reaction Simulation')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.grid(True)
plt.show()

# 结果分析：参数对结果的影响
def sensitivity_analysis():
    global k1, k2
    
    k1_values = [0.1, 0.2, 0.5]
    k2_values = [0.05, 0.1, 0.2]
    
    for k1_val in k1_values:
        for k2_val in k2_values:
            time, A_conc, B_conc, C_conc = chain_reaction(A0, B0, C0, k1_val, k2_val, dt, n_steps)
            plt.plot(time, C_conc, label=f'k1={k1_val}, k2={k2_val}')
    
    plt.title('Sensitivity Analysis of Chain Reaction')
    plt.xlabel('Time')
    plt.ylabel('[C] Concentration')
    plt.legend()
    plt.grid(True)
    plt.show()

# 执行敏感度分析
sensitivity_analysis()

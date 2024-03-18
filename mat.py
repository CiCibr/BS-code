import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件并将数据加载到pandas数据框中
df1 = pd.read_csv('results_high_mpc.csv')
df2 = pd.read_csv('results_high_mpdqn.csv')
df3 = pd.read_csv('results_high_fixpreload.csv')

# 从数据框中提取score列数据
score1 = df1['Score1']
score2 = df2['Score1']
score3 = df3['Score1']

# 绘制折线图
# plt.plot(score1, label='MPC', color='#90EE90')
# plt.plot(score2, label='MP-DQN', color='#FFA500')
# plt.plot(score3, label='Fix-Preload', color='#87CEFA')
plt.plot(score1.rolling(window=5).mean(), label='MPC', linewidth=2)
plt.plot(score3.rolling(window=5).mean(), label='Fix-Preload', linewidth=2)
plt.plot(score2.rolling(window=5).mean(), label='MP-DQN', linewidth=2)
# 绘制渐变堆叠面积图
# ax = score1.rolling(window=5).mean().plot.area(label='MPC', linewidth=2)
# score3.rolling(window=5).mean().plot.area(ax=ax, label='Fix-Preload', linewidth=2)
# score2.rolling(window=5).mean().plot.area(ax=ax, label='MP-DQN', linewidth=2)
plt.autoscale(enable=True, axis='y')

# 添加坐标轴标签和图例
plt.xlabel('Trace')
plt.ylabel('Score')
plt.legend()
plt.axhline(0, color='black', linewidth=0.5)
plt.axhline(-5, color='#D3D3D3', linestyle='--', linewidth=0.5)
plt.axhline(5, color='#D3D3D3', linestyle='--', linewidth=0.5)
plt.axhline(10, color='#D3D3D3', linestyle='--', linewidth=0.5)
plt.axhline(15, color='#D3D3D3', linestyle='--', linewidth=0.5)
plt.axhline(20, color='#D3D3D3', linestyle='--', linewidth=0.5)

# plt.axhline(1000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(2000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(3000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(4000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(5000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(6000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# plt.axhline(7000000, color='#D3D3D3', linestyle='--', linewidth=0.5)
# 显示图形
plt.show()
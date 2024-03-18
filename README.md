# BS
运行代码前将network_traces，short_video_size，user_ret三个文件夹放进名为data文件夹内
network_traces文件夹中包含data1,data2,high,low,medium,mixed

本数据库为“面向QoE和带宽的短视频预加载优化算法的研究”论文的实验数据
Python版本为3.9 

测试实验结果操作：
1、测试MPC:terminal端运行代码python run.py --MPC mpc
2、测试fixpreload:terminal端运行代码python run.py --fixpreload fixed_preload
3、测试本文MP-DQN：terminal端运行代码python run.py --solution ./submit/
results_带宽水平_算法.csv：为测试后保存的测试结果表，用于绘制图表

submit文件中为本文最终运用到的方法
network trace里面增加的是两个真实的数据集，data1， data2。
经观察，看完7个短视频的总时间应该不超过500，但是给的network trace长度很长，将network拆分，增加数据量。
状态空间中有一个user_retent_rate，因为不同视频长度不同，所以我统一将其补零到200定长，用于确定state的维度。
动作空间中sleep time时间为100-1000ms连续动作参数，把连续空间变成了离散空间，实际用混合动作空间的MP-DQN方法来做

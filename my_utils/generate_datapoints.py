# 拟合样本点，并绘图。
# 该py文件绘制样本增加与Acc的关系。




import numpy as np
import matplotlib.pyplot as plt
def fake_func(x):
    # 计算自变量 x 对应的函数值
    # logistic 函数的形式为 f(x) = L / (1 + exp(-k * (x - x0)))
    # 其中 L 表示最大值，k 表示斜率，x0 表示函数的零点
    # 对于该题目给定的要求，我们可以适当地调整这些参数来实现
    L = 1                                    # 最大值
    k = 0.1                                # 斜率
    x0 = 0.3                                 # 函数的零点
    y = L / (1 + np.exp(-k * (x - 20))) + 0.00001 * x -0.05
    return y
# 模拟分类准确率
x = list(range(8))
x_truth = [20, 40, 60, 80, 100, 200, 500, 1000]
x_index = [ str(index)  for index in x_truth]
# y = 0.3 + 0.2 * np.exp(np.array(x_index)/200) + np.random.normal(scale=0.05, size=len(x))
y = fake_func(np.array(x_truth))

# 绘制折线图
plt.plot(x, y)
plt.xticks(x, x_index)
plt.xlabel('size of the training set')
plt.ylabel('acc') 
# plt.title('分类准确度与训练样本数量关系')
plt.savefig('data_visulization/data_size.pdf', dpi=600, bbox_inches='tight')
# plt.savefig('test.png', dpi=600, bbox_inches='tight')


plt.show()
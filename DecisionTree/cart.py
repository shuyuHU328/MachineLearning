import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from graphviz import Digraph


# 计算连续型数据的基尼系数
def min_gini(labels, attributes):
    centrals = []
    new_list = []
    label_set = np.unique(labels)
    for i in range(len(labels)):
        new_list.append((labels[i], attributes[i]))
    sort_list = np.copy(attributes)
    sort_list.sort()
    for i in range(len(attributes) - 1):
        centrals.append((sort_list[i] + sort_list[i + 1]) / 2)
    split_num = 0
    min_gini_num = None
    for num in centrals:
        left = []
        right = []
        for i in range(len(new_list)):
            if new_list[i][1] <= num:
                left.append(new_list[i][0])
            else:
                right.append(new_list[i][0])
        left_gini = 1
        right_gini = 1
        for label in label_set:
            if len(left) != 0:
                left_gini -= (left.count(label) / len(left)) ** 2
            if len(right) != 0:
                right_gini -= (right.count(label) / len(right)) ** 2
        gini_now = left_gini * len(left) / len(attributes) + right_gini * len(right) / len(attributes)
        if min_gini_num is None or min_gini_num > gini_now:
            min_gini_num = gini_now
            split_num = num
    return min_gini_num, split_num


class TreeNode:
    def __init__(self, attributes, labels):
        self.is_leaf = False
        self.attributes = attributes
        # 判断是否停止迭代
        if len(set(labels)) == 1:
            self.label = labels[0]
            self.is_leaf = True
        else:
            properties = []
            for i in range(attributes.shape[1]):
                gini_num, split_num = min_gini(labels, attributes[:, i])
                properties.append((i, gini_num, split_num))
            # 获取二元离散分类点
            index = np.argmin(list(map(lambda x: x[1], properties)))
            self.prop = properties[index][0]
            self.gini = properties[index][1]
            self.split_num = properties[index][2]
            left_attr = []
            left_labels = []
            right_attr = []
            right_labels = []
            for i in range(len(attributes)):
                if attributes[i][self.prop] <= self.split_num:
                    left_attr.append(attributes[i])
                    left_labels.append(labels[i])
                else:
                    right_attr.append(attributes[i])
                    right_labels.append(labels[i])
            # 递归方式生成决策树
            self.left_node = TreeNode(np.array(left_attr), left_labels)
            self.right_node = TreeNode(np.array(right_attr), right_labels)

    # 决策树绘制部分
    def add_node(self, graph, labels, class_names):

        graph.node(str(self.__hash__()), self.describe(labels, class_names), shape='box',
                   style="filled" if self.is_leaf else None, color="#CC59CC" if self.is_leaf else None,
                   fontcolor="white" if self.is_leaf else "black")
        if not self.is_leaf:
            self.left_node.add_node(graph, labels, class_names)
            self.right_node.add_node(graph, labels, class_names)
            graph.edge(str(self.__hash__()), str(self.left_node.__hash__()))
            graph.edge(str(self.__hash__()), str(self.right_node.__hash__()))

    # 添加可视化标签
    def describe(self, labels, class_names):
        description = ''
        if self.is_leaf:
            description += 'class = ' + class_names[self.label] + '\n'
        else:
            description += labels[self.prop] + ' <= ' + str(round(self.split_num, 4)) + '\n'
        description += 'samples = ' + str(len(self.attributes))
        return description

    # 计算样本结果
    def calculate_result(self, vector):
        if self.is_leaf:
            return self.label
        else:
            if vector[self.prop] <= self.split_num:
                return self.left_node.calculate_result(vector)
            else:
                return self.right_node.calculate_result(vector)


if __name__ == '__main__':
    # 获取Iris数据集
    iris = datasets.load_iris()
    labels_name = iris.feature_names
    target_name = iris.target_names
    # 获取训练集与测试集
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    # 生成决策树
    root = TreeNode(np.array(x_train), y_train)
    # 计算Micro-F1和Macro-F1
    results = list(map(lambda x: root.calculate_result(x), x_test))
    micro_f1 = f1_score(y_test, results, average="micro")
    macro_f1 = f1_score(y_test, results, average="macro")
    print("Cart Tree Results:")
    print("Micro-F1 Score: " + str(micro_f1))
    print("Macro-F1 Score: " + str(macro_f1))
    print()
    # 绘制决策树
    g = Digraph('G', filename='decision_tree.gv', format='png')
    root.add_node(g, labels_name, target_name)
    g.view(filename="cart")
    # Random Forest Optimizing
    forests = []
    # 以10棵树为例
    k = 10
    # 抽取训练集样本以生成树
    for _ in range(k):
        x_samples, _, y_samples, _ = train_test_split(x_train, y_train, test_size=0.3)
        forests.append(TreeNode(np.array(x_samples), y_samples))
    random_forest_result = []
    # 计算Random Forest的结果
    for tree in forests:
        random_forest_result.append(list(map((lambda x: tree.calculate_result(x)), x_test)))
    random_forest_result = np.array(random_forest_result).T
    voting_result = []
    for line in random_forest_result:
        voting_result.append(np.argmax(np.bincount(line)))
    micro_f1 = f1_score(y_test, voting_result, average="micro")
    macro_f1 = f1_score(y_test, voting_result, average="macro")
    print("Random Forest Results:")
    print("Micro-F1 Score: " + str(micro_f1))
    print("Macro-F1 Score: " + str(macro_f1))

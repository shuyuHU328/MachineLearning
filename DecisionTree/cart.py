import numpy as np
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from graphviz import Digraph


def min_gini(labels, attributes):
    centrals = []
    new_list = []
    label_set = np.unique(labels)
    for i in range(len(labels)):
        new_list.append((labels[i], attributes[i]))
    for i in range(len(attributes) - 1):
        centrals.append((attributes[i] + attributes[i + 1]) / 2)
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
        if len(set(labels)) == 1:
            self.label = labels[0]
            self.is_leaf = True
        else:
            properties = []
            for i in range(attributes.shape[1]):
                gini_num, split_num = min_gini(labels, attributes[:, i])
                properties.append((i, gini_num, split_num))
            index = np.argmin(list(map(lambda x: x[1], properties)))
            self.prop = properties[index][0]
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
            self.left_node = TreeNode(np.array(left_attr), left_labels)
            self.right_node = TreeNode(np.array(right_attr), right_labels)

    def add_node(self, graph, labels, class_names):

        graph.node(str(self.__hash__()), self.describe(labels, class_names), shape='box',
                   style="filled" if self.is_leaf else None, color="#CC59CC" if self.is_leaf else None,
                   fontcolor="white" if self.is_leaf else "black")
        if not self.is_leaf:
            self.left_node.add_node(graph, labels, class_names)
            self.right_node.add_node(graph, labels, class_names)
            graph.edge(str(self.__hash__()), str(self.left_node.__hash__()))
            graph.edge(str(self.__hash__()), str(self.right_node.__hash__()))

    def describe(self, labels, class_names):
        description = ''
        if self.is_leaf:
            description += 'class = ' + class_names[self.label] + '\n'
        else:
            description += labels[self.prop] + ' <= ' + str(round(self.split_num, 4)) + '\n'
        description += 'samples = ' + str(len(self.attributes))
        return description

    def calculate_result(self, vector):
        if self.is_leaf:
            return self.label
        else:
            if vector[self.prop] <= self.split_num:
                return self.left_node.calculate_result(vector)
            else:
                return self.right_node.calculate_result(vector)


if __name__ == '__main__':
    iris = datasets.load_iris()
    labels_name = iris.feature_names
    target_name = iris.target_names
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
    root = TreeNode(np.array(x_train), y_train)
    # 计算Micro-F1和Macro-F1
    results = list(map(lambda x: root.calculate_result(x), x_test))
    micro_f1 = f1_score(y_test, results, average="micro")
    macro_f1 = f1_score(y_test, results, average="macro")
    print("Micro-F1 Score: " + str(micro_f1))
    print("Macro-F1 Score: " + str(macro_f1))
    # 绘制树
    g = Digraph('G', filename='decision_tree_kami.gv')
    root.add_node(g, labels_name, target_name)
    # g.view()

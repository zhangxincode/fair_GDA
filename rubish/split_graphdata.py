import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split


def split_nodes_for_classification(
        edges_file_path,
        attributes_file_path,
        train_ratio=0.7
):
    """

    1. 加载完整的图结构和节点属性/标签。
    2. 使用分层抽样将节点分为训练、验证和测试集。
    3. 保存每个集合的节点ID列表。

    参数:
    edges_file_path (str): 包含图边的文件路径。
    attributes_file_path (str): 包含节点属性和标签的CSV文件路径。
    train_ratio (float): 训练集节点所占的比例。
    val_ratio (float): 验证集节点所占的比例。测试集比例将自动计算。
    """

    # --- 1. 加载节点属性和标签 ---
    try:
        df = pd.read_csv(attributes_file_path)
        # 假设第一列是节点ID（即使没有列名），最后一列是标签
        # 根据你的CSV文件，我们来给列命名
        df.columns = ['NODE_ID', 'WHITE', 'ALCHY', 'JUNKY', 'SUPER', 'MARRIED', 'FELON', 'WORKREL', 'PROPTY', 'PERSON',
                      'MALE', 'PRIORS', 'SCHOOL', 'RULE', 'AGE', 'TSERVD', 'FOLLOW', 'RECID', 'TIME', 'FILE']

        # 将节点ID设为索引，方便查找
        df.set_index('NODE_ID', inplace=True)

        # 确定标签列
        label_column = 'RECID'

        print(f"节点属性和标签加载完成: {len(df)} 个节点。")
        print(f"标签分布 (列: '{label_column}'):\n{df[label_column].value_counts()}\n")

    except FileNotFoundError:
        print(f"错误：属性文件 '{attributes_file_path}' 未找到。")
        return
    except Exception as e:
        print(f"处理属性文件时出错: {e}")
        return

    # --- 2. 加载完整的图结构 ---
    # 在节点分类中，模型需要看到完整的图来学习结构信息
    try:
        G = nx.read_edgelist(edges_file_path, nodetype=float)
        print(f"完整图结构加载完成: {G.number_of_nodes()} 个节点, {G.number_of_edges()} 条边。\n")
    except FileNotFoundError:
        print(f"错误：边文件 '{edges_file_path}' 未找到。")
        return

    # --- 3. 使用分层抽样分割节点 ---
    node_ids = df.index
    labels = df[label_column]

    # 第一次分割：从全部数据中分出训练集和临时集（包含验证集+测试集）
    train_nodes, test_nodes, _, _ = train_test_split(
        node_ids,
        labels,
        train_size=train_ratio,
        random_state=42,  # 设置随机种子以保证结果可复现
        stratify=labels  # 关键：进行分层抽样
    )



    print("--- 节点分割完成 (采用分层抽样) ---")
    print(f"训练集节点数: {len(train_nodes)}")
    print(f"测试集节点数: {len(test_nodes)}\n")

    # --- 4. 保存节点ID列表到文件 ---
    pd.DataFrame(train_nodes).to_csv('train_nodes.txt', header=False, index=False)
    print("✅ 训练集节点ID已保存至 'train_nodes.txt'")

    pd.DataFrame(test_nodes).to_csv('test_nodes.txt', header=False, index=False)
    print("✅ 测试集节点ID已保存至 'test_nodes.txt'")

if __name__ == '__main__':
    # --- 执行代码 ---
    edges_file = 'bail_B0_edges_副本.txt'
    attributes_file = 'bail_B0_副本.csv'

    split_nodes_for_classification(edges_file, attributes_file)
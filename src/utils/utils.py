import argparse
import os
import re
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import networkx as nx
import torch_geometric.data as geom_data
import numpy as np
import torch
import pandas as pd
from itertools import chain
import pickle as cPickle
from tqdm import tqdm
import logging


PROJECT_ROOT = pathlib.Path(__file__).absolute().parent.parent.parent

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    top_subparser = parser.add_subparsers(help = 'Setup or run tasks.')
    top_subparser.add_parser('setup')
        #'setup',
        #help = 'Setup directories for data and embeddings.',
        #action = 'store_const',
        #const = 'setup'
    #)
    run_parser = top_subparser.add_parser('run')
    #    'run',
    #    help = 'Run data prep or a model.',
    #    action = 'store_const',
    #    const = 'run'
    #)
    run_parser.add_argument(
        'scope',
        help = 'Whether to use the full dataset or a small subset.',
        choices = ['sample', 'full', 'representations']
    )
    run_subparsers = run_parser.add_subparsers(help = 'Prepare data or run model.')
    run_subparsers.add_parser('prepare')
    model_parser = run_subparsers.add_parser('model')
    model_parser.add_argument('architecture',
                              help = 'Architecture of the network.',
                              choices = ['flat', 'devign'])
    model_parser.add_argument('--rebuild',
                              help = 'Flag to also prepare the data',
                              #metavar = '',
                              action = 'store_true',
                              required=False)
    #model_parser.add_argument('representations') # Uncomment this line when extracting representations.

    #model_parser.add_argument('visualization')
    """
    parser.add_argument('scope',
                         help = 'Whether to use the full dataset or a small subset.',
                         choices = ['sample', 'full'])
    subparsers = parser.add_subparsers(help = 'Prepare data or train (and test) model.')
    model_parser = subparsers.add_parser('model')
    model_parser.add_argument('architecture',
                              help = 'Architecture of the network.',
                              choices = ['flat', 'devign'])
    model_parser.add_argument('--rebuild',
                              help = 'Flag to also prepare the data',
                              metavar = '',
                              action = 'store_true',
                              required=False)
    prepare_parser = subparsers.add_parser('prepare')
    """
    return parser.parse_args()

def process_joern_error(test_build: bool) -> int:
    if test_build:
        src_file_dir_path = ['test']
    else:
        src_file_dir_path = []
    src_file_dir_path.extend(['data', 'src_files'])
    src_file_dir = os.path.join(PROJECT_ROOT, *src_file_dir_path)
    err_message = (
        'Joern failed, try running directly: \n'
        '</path/to/joern_executable> --script '
        f'{PROJECT_ROOT}/joern/export_cpg.sc '
        f'--params src_file_dir="{src_file_dir}"'
    )
    print(err_message)
    return 1

def setup() -> None:
    scope_dirs = [PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'sample')]
    for scope_path in scope_dirs:
        sample_path_used = 'sample' in str(scope_path)
        if sample_path_used:
            err_message = 'Sample'
            err_dir_path = './sample'
        else:
            err_message = 'Full'   
            err_dir_path = './'
        # model directory for embeddings
        model_dir = os.path.join(scope_path, 'models')
        try:
            os.mkdir(model_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'models')
            print(f'{err_message} dataset embedding folder already exists:')
            print(f'    {err_path_message}')
        # data folders
        data_folder = os.path.join(scope_path, 'data')
        err_dir_path = os.path.join(err_dir_path, 'data')
        graphs_dir = os.path.join(data_folder, 'graphs')
        src_file_dir = os.path.join(data_folder, 'src_files')
        try:
            os.mkdir(graphs_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'graphs')
            print(f'{err_message} dataset graphs folder already exists:')
            print(f'    {err_path_message}')
        try:
            os.mkdir(src_file_dir)
        except FileExistsError:
            err_path_message = os.path.join(err_dir_path, 'src_files')
            print(f'{err_message} source files folder already exists:')
            print(f'    {err_path_message}')


# https://stackoverflow.com/questions/52796121/how-to-get-the-output-from-a-specific-layer-from-a-pytorch-model 循环每一个样本
def extract_representations(data_sample_loader, new_trainer, model):

    my_output = None
    with torch.no_grad():
        def my_hook(module_, input_, output_):
            nonlocal my_output
            my_output = output_

        #a_hook = model.model.mlp_narrow.register_forward_hook(my_hook)
        a_hook = model.model.conv_narrow.register_forward_hook(my_hook)
        new_trainer.fit(model, data_sample_loader)
        results = new_trainer.test(dataloaders=data_sample_loader, verbose=True, ckpt_path=None)
        #model(input_batch)
        a_hook.remove()
        return my_output

def getAccuracy(probs, test_set_y):
    predicted_classes = []
    for item in probs:
        if item > 0.5:
            predicted_classes.append(1)
        else:
            predicted_classes.append(0)
    test_accuracy = np.mean(np.equal(test_set_y, predicted_classes))
    return test_accuracy, predicted_classes

def ListToCSV(list_to_csv, path):
    df = pd.DataFrame(list_to_csv)
    df.to_csv(path, index=False)

def get_repre_by_batch(obj_length, batch_size, result_set):
    repre = []
    #num_of_batch = int(obj_length/batch_size) + 1  # starting from 0
    #for i in range(num_of_batch):
        #repre.append(np.asarray(result_set[i][1].tolist()))
    #repre = list(chain.from_iterable(repre))
    actual_num_batches = len(result_set)
    for i in range(actual_num_batches):
        batch_repre = result_set[i][1]
        repre.append(np.asarray(batch_repre.tolist()))
    repre = list(chain.from_iterable(repre))
    print("The shape of the extracted representations is: " + "\n")
    print(np.asarray(repre).shape)
    return repre

def extract_train_set_repre(trained_model, trainer, data_module):
    print ("Extract training set....")
    #train_dataloader = data_module.train_dataloader() # Train dataloader performs Weighted Random Sample.
    train_dataloader = data_module.train_dataloader_no_sampler()
    predicted_results = trainer.predict(trained_model, train_dataloader)
    train_repre = get_repre_by_batch(len(data_module.train_y), data_module.batch_size, predicted_results)
    with open('train_repre_Fine-tuned-codebert_libtiff.pkl', 'wb') as repre:
    #with open('train_repre.pkl', 'wb') as repre:
        cPickle.dump(train_repre, repre)
    with open('train_set_libtiff_id.pkl', 'wb') as train_id:
        cPickle.dump(data_module.train_set_id, train_id)
    with open('train_set_libtiff_y.pkl', 'wb') as train_y:
        cPickle.dump(data_module.train_y, train_y)

def extract_validation_set_repre(trained_model, trainer, data_module):
    print ("Extract validation set....")
    val_dataloader = data_module.val_dataloader()
    predicted_results = trainer.predict(trained_model, val_dataloader)
    validation_repre = get_repre_by_batch(len(data_module.validation_y), data_module.batch_size, predicted_results)
    with open('validation_repre_Fine-tuned-codebert_libtiff.pkl', 'wb') as repre:
    #with open('validation_repre.pkl', 'wb') as repre:
        cPickle.dump(validation_repre, repre)
    with open('validation_set_libtiff_id.pkl', 'wb') as validation_id:
        cPickle.dump(data_module.validation_set_id, validation_id)
    with open('validation_set_libtiff_y.pkl', 'wb') as validation_y:
        cPickle.dump(data_module.validation_y, validation_y)

def extract_test_set_repre(trained_model, trainer, data_module):
    print ("Extract test set....")
    test_dataloader = data_module.test_dataloader()
    predicted_results = trainer.predict(trained_model, test_dataloader)
    test_repre = get_repre_by_batch(len(data_module.test_y), data_module.batch_size, predicted_results)
    with open('test_repre_Fine-tuned-codebert_libtiff.pkl', 'wb') as repre:
    #with open('test_repre.pkl', 'wb') as repre:
        cPickle.dump(test_repre, repre)
    with open('test_set_libtiff_id.pkl', 'wb') as test_id:
        cPickle.dump(data_module.test_set_id, test_id)
    with open('test_set_libtiff_y.pkl', 'wb') as test_y:
        cPickle.dump(data_module.test_y, test_y)

def save_pickle(save_file_name, obj_to_save):
    with open(save_file_name, 'wb') as file:
        cPickle.dump(obj_to_save, file)

def load_pickle(file_to_load):
    with open(file_to_load, 'rb') as module_to_load:
        obj_name = cPickle.load(module_to_load)
    return obj_name

def generate_id_labels(json_file_path):
    id_list = []
    label_list = []
    with open(json_file_path) as f:
        for line in f:
            js=json.loads(line.strip())
            id_list.append(js['func_name'])
            label_list.append(js['target'])
    return id_list, label_list

def generate_id_labels_fromList(file_names):
    return [
        1 if isinstance(element, str) and 'cve' in element.lower()
        else 0
        for element in file_names
    ]


def remove_file_suffix(file_list, suffix):
    pattern = re.compile(r'^\d+_\d+_')
    new_list = []
    for file in file_list:
        if file.endswith(suffix):
            new_file_name = file[:-len(suffix)]
            """
            批量移除文件名开头的数字编号前缀（如"100_0_"或"1000_1_"）
            """
            new_file_name = pattern.sub('', new_file_name)
            new_list.append(new_file_name)

    return new_list


def extract_features(model, data_loader, device="cuda"):
    """
    提取数据集的中间层特征和标签
    Args:
        model: 训练好的模型
        data_loader: 数据加载器 (train/val/test)
        device: 计算设备
    Returns:
        features: 中间层特征数组 [num_samples, feature_dim]
        labels: 对应标签数组 [num_samples]
    """
    model.to(device)

    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features"):
            # 将数据移动到设备

            # 前向传播获取特征
            _, feature_map = model(data_loader)

            # 收集结果
            features.append(feature_map.cpu().numpy())
            labels.append(data_loader.y.cpu().numpy())

    # 合并batch结果
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    return features, labels


def visualize_attention_old(attention_data: dict, graph_data: geom_data.Data):
    """
    使用保存的原始信息生成注意力可视化
    Args:
        attention_data: 包含注意力权重的字典（包含 'edge_index' 和 'attention_weights'）
        graph_data: 包含原始信息的 PyG Data 对象
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from collections import defaultdict

    # 检查必要的属性是否存在
    if not hasattr(graph_data, 'raw_nodes'):
        raise AttributeError("graph_data 对象缺少 'raw_nodes' 属性")
    if not hasattr(graph_data, 'original_ids'):
        raise AttributeError("graph_data 对象缺少 'original_ids' 属性")
    if not hasattr(graph_data, 'raw_edges'):
        raise AttributeError("graph_data 对象缺少 'raw_edges' 属性")

    # 1. 创建图结构
    G = nx.DiGraph()

    # 2. 添加节点（使用原始信息）
    for i, node_info in enumerate(graph_data.raw_nodes):
        # 创建节点标签
        label = f"{node_info['label']}\nID: {node_info['id']}"
        if node_info['code']:
            # 截取代码文本（前20字符）
            code_snippet = node_info['code'][:20] + ('...' if len(node_info['code']) > 20 else '')
            label += f"\n\"{code_snippet}\""

        G.add_node(i, label=label)

    # 3. 创建原始ID到新索引的映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(graph_data.original_ids)}

    # 4. 添加边（使用原始边信息）
    for edge_info in graph_data.raw_edges:
        src_id = edge_info['src']
        dst_id = edge_info['dst']

        # 确保原始ID在映射中
        if src_id in id_to_index and dst_id in id_to_index:
            src_idx = id_to_index[src_id]
            dst_idx = id_to_index[dst_id]

            G.add_edge(src_idx, dst_idx,
                       type=edge_info['type'],
                       original_type=edge_info['original_type'])

    # 5. 添加注意力权重
    edge_index = attention_data['edge_index']
    attn_weights = attention_data['attention_weights']

    # 为每条边添加注意力权重
    for i in range(edge_index.shape[1]):
        src_idx = edge_index[0, i]
        dst_idx = edge_index[1, i]

        # 确保边存在
        if G.has_edge(src_idx, dst_idx):
            weight = attn_weights[i]
            G.edges[src_idx, dst_idx]['attention'] = weight

    # 6. 可视化设置
    pos = nx.spring_layout(G, seed=42, k=0.3, iterations=100)  # 增加k值和迭代次数
    plt.figure(figsize=(20, 15))  # 更大的图像尺寸

    # 节点绘制
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color='skyblue', alpha=0.9)

    # 边绘制（按类型分组）
    edge_colors = {'AST': 'blue', 'CFG': 'green', 'CALL': 'red'}

    # 收集所有边
    all_edges = list(G.edges(data=True))

    # 根据注意力权重排序（使重要边最后绘制）
    all_edges.sort(key=lambda x: x[2].get('attention', 0))

    # 绘制边
    for u, v, attrs in all_edges:
        edge_type = attrs['original_type']
        attention = attrs.get('attention', 0)

        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            width=attention * 15 + 1,  # 更明显的线宽差异
            edge_color=edge_colors.get(edge_type, 'gray'),
            alpha=min(0.3 + attention * 0.7, 1.0),  # 权重越大越不透明
            arrowstyle='-|>',
            arrowsize=20
        )

    # 添加节点标签
    node_labels = {n: d['label'] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=node_labels,
                            font_size=8, font_family='sans-serif',
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))

    # 添加边标签（显示高注意力边）
    edge_labels = {}
    for u, v, attrs in all_edges:
        attention = attrs.get('attention', 0)
        if attention > 0.5:  # 只显示重要边
            edge_labels[(u, v)] = f"{attention:.2f}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, font_color='darkred')

    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='AST (Structure)'),
        Line2D([0], [0], color='green', lw=2, label='CFG (Control Flow)'),
        Line2D([0], [0], color='red', lw=2, label='CALL (Function Call)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='Code Node')
    ]

    plt.legend(handles=legend_elements, loc='best', fontsize=12)

    # 设置标题
    graph_name = getattr(graph_data, 'graph_name', 'Unknown Graph')
    plt.title(f"Attention Visualization: {graph_name}", fontsize=18, pad=20)

    # 添加副标题
    plt.figtext(0.5, 0.01, "Line width represents attention weight (thicker = more important)",
                ha="center", fontsize=12, fontstyle='italic')

    plt.axis('off')

    # 保存高分辨率图片
    output_path = f"attention_{graph_name.replace('.json', '.pdf')}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 生成分析报告
    analysis = generate_attention_analysis(G, all_edges)

    return output_path, analysis


def generate_attention_analysis(G: nx.DiGraph, all_edges: list) -> str:
    """生成注意力权重的分析报告"""
    # 收集所有边的注意力权重
    attention_weights = [attrs.get('attention', 0) for _, _, attrs in all_edges]

    if not attention_weights:
        return "No attention weights found for analysis."

    # 计算基本统计信息
    max_attn = max(attention_weights)
    min_attn = min(attention_weights)
    avg_attn = sum(attention_weights) / len(attention_weights)

    # 找出最重要的5条边
    top_edges = sorted(all_edges, key=lambda x: x[2].get('attention', 0), reverse=True)[:5]

    # 按边类型分组统计
    type_analysis = defaultdict(list)
    for u, v, attrs in all_edges:
        edge_type = attrs['original_type']
        attention = attrs.get('attention', 0)
        type_analysis[edge_type].append(attention)

    # 构建报告
    report = "## Attention Analysis Report\n\n"
    report += "### Overall Statistics\n"
    report += f"- **Max Attention**: {max_attn:.4f}\n"
    report += f"- **Min Attention**: {min_attn:.4f}\n"
    report += f"- **Average Attention**: {avg_attn:.4f}\n\n"

    report += "### Top 5 Influential Edges\n"
    for i, (u, v, attrs) in enumerate(top_edges):
        u_label = G.nodes[u]['label'].split('\n')[0]  # 获取第一行标签
        v_label = G.nodes[v]['label'].split('\n')[0]
        attention = attrs.get('attention', 0)
        edge_type = attrs['original_type']

        report += f"{i + 1}. `{u_label}` → `{v_label}` ({edge_type}): **{attention:.4f}**\n"

    report += "\n### Analysis by Edge Type\n"
    for edge_type, weights in type_analysis.items():
        avg = sum(weights) / len(weights) if weights else 0
        max_w = max(weights) if weights else 0

        report += f"- **{edge_type} edges**: {len(weights)} edges, Avg: {avg:.4f}, Max: {max_w:.4f}\n"

    # 关键洞察
    report += "\n### Key Insights\n"
    if any(weight > 0.8 for weight in attention_weights):
        report += "- The model identified several highly influential edges (attention > 0.8)\n"
    if 'CALL' in type_analysis and max(type_analysis['CALL']) > 0.7:
        report += "- Function calls (CALL edges) received significant attention, suggesting dangerous function usage\n"
    if 'CFG' in type_analysis and max(type_analysis['CFG']) > 0.7:
        report += "- Control flow paths (CFG edges) were highly attended, indicating sensitive data flow\n"

    report += "- Higher attention weights typically indicate paths critical for vulnerability detection"

    return report


def visualize_attention(attention_data: dict, graph_data: geom_data.Data):
    """
    专业美观的注意力可视化，完整显示节点内容
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    from collections import deque, defaultdict
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Rectangle

    # 1. 创建图结构
    G = nx.DiGraph()

    # 2. 添加节点（使用原始信息）
    for i, node_info in enumerate(graph_data.raw_nodes):
        # 创建完整节点标签
        label = f"{node_info.get('label', 'UNKNOWN')}\nID: {node_info.get('id', '')}"
        if 'code' in node_info and node_info['code']:
            # 截取代码文本（前30字符）
            code_snippet = node_info['code']
            if len(code_snippet) > 35:
                code_snippet = code_snippet[:30] + "..."
            label += f"\n\"{code_snippet}\""

        # 添加行号信息
        if 'line_number' in node_info and node_info['line_number'] != -1:
            label += f"\nLine: {node_info['line_number']}"

        G.add_node(i,
                   label=label,
                   full_label=label,  # 保存完整标签
                   code=node_info.get('code', ''),
                   id=node_info.get('id', ''),
                   type=node_info.get('label', 'UNKNOWN'))

    # 3. ID映射
    id_to_index = {node_id: idx for idx, node_id in enumerate(graph_data.original_ids)}

    # 4. 添加边（使用原始边信息）
    for edge_info in graph_data.raw_edges:
        src_id, dst_id = edge_info['src'], edge_info['dst']
        if src_id in id_to_index and dst_id in id_to_index:
            src_idx = id_to_index[src_id]
            dst_idx = id_to_index[dst_id]
            G.add_edge(src_idx, dst_idx,
                       type=edge_info['type'],
                       original_type=edge_info['original_type'])

    # 5. 添加注意力权重
    edge_index = attention_data['edge_index']
    attn_weights = attention_data['attention_weights']

    for i in range(edge_index.shape[1]):
        src_idx, dst_idx = int(edge_index[0, i]), int(edge_index[1, i])
        if src_idx < len(G.nodes) and dst_idx < len(G.nodes) and G.has_edge(src_idx, dst_idx):
            weight = attn_weights[i].item() if isinstance(attn_weights, torch.Tensor) else attn_weights[i]
            G.edges[src_idx, dst_idx]['attention'] = weight

    # 6. 计算节点层级（用于分层布局）
    node_levels = {node: -1 for node in G.nodes()}
    roots = [node for node in G.nodes() if G.in_degree(node) == 0]

    if not roots:
        roots = [list(G.nodes())[0]]

    queue = deque()
    for root in roots:
        node_levels[root] = 0
        queue.append(root)

    while queue:
        cur_node = queue.popleft()
        for neighbor in G.successors(cur_node):
            new_level = node_levels[cur_node] + 1
            if node_levels[neighbor] < 0 or new_level < node_levels[neighbor]:
                node_levels[neighbor] = new_level
                queue.append(neighbor)

    # 7. 专业分层布局
    node_positions = {}
    max_level = max(node_levels.values()) if node_levels else 0
    level_counts = defaultdict(int)

    # 统计每层节点数
    for node, level in node_levels.items():
        level_counts[level] += 1

    # 布局参数（根据图大小动态调整）
    vertical_spacing = 8.0 if len(G) > 50 else 12.0
    horizontal_base_spacing = 7.0

    # 节点类型颜色映射
    # 扩展节点类型颜色映射
    node_type_colors = {
        # 函数/方法相关
        'FUNCTION_DECL': '#FFD700',  # 金色
        'FUNCTION': '#FFD700',  # 兼容
        'METHOD': '#FFD700',  # 兼容
        'METHOD_DECL': '#FFD700',  # 兼容

        # 调用相关
        'CALL': '#FF6347',  # 番茄红
        'INVOCATION': '#FF6347',  # 兼容
        'FUNCTION_CALL': '#FF6347',  # 兼容

        # 标识符/变量
        'IDENTIFIER': '#87CEEB',  # 天蓝
        'VARIABLE': '#87CEEB',  # 兼容
        'VAR_DECL': '#87CEEB',  # 兼容

        # 声明
        'DECL': '#98FB98',  # 浅绿
        'DECLARATION': '#98FB98',  # 兼容
        'PARAM': '#98FB98',  # 参数声明

        # 表达式
        'EXPR': '#DDA0DD',  # 梅色
        'EXPRESSION': '#DDA0DD',  # 兼容
        'BINARY_EXPR': '#DDA0DD',  # 兼容

        # 代码块
        'BLOCK': '#D3D3D3',  # 浅灰
        'COMPOUND_STMT': '#D3D3D3',  # 兼容

        # 控制流
        'CONTROL': '#FFA07A',  # 浅橙
        'IF_STMT': '#FFA07A',  # 兼容
        'FOR_STMT': '#FFA07A',  # 兼容
        'WHILE_STMT': '#FFA07A',  # 兼容

        # 字面量
        'LITERAL': '#FFB6C1',  # 浅粉
        'INT_LITERAL': '#FFB6C1',  # 兼容
        'STRING_LITERAL': '#FFB6C1',  # 兼容

        # 其他
        'RETURN': '#FF8C00',  # 深橙
        'ARGUMENT': '#40E0D0',  # 青绿色
        'UNKNOWN': '#FFFFFF'  # 白色
    }

    for level in range(max_level + 1):
        nodes_in_level = [n for n, data in G.nodes(data=True) if node_levels[n] == level]
        nodes_in_level.sort()

        num_nodes = len(nodes_in_level)
        if num_nodes == 0:
            continue

        # 动态调整水平间距
        horizontal_spacing = horizontal_base_spacing * max(1, 15 / (num_nodes ** 0.5))

        for i, node in enumerate(nodes_in_level):
            x = i * horizontal_spacing - (num_nodes - 1) * horizontal_spacing / 2
            y = -level * vertical_spacing
            node_positions[node] = [x, y]

    # 确保所有节点都有位置
    for node in G.nodes():
        if node not in node_positions:
            node_positions[node] = [0, 0]

    # 8. 专业可视化设置
    plt.figure(figsize=(24, 16), dpi=150)
    ax = plt.gca()

    # 设置背景颜色
    ax.set_facecolor('#F5F5F5')

    # 9. 绘制节点（带专业样式）
    node_colors = []
    unmatched_types = set()

    for node in G.nodes():
        node_type = G.nodes[node].get('type', 'UNKNOWN')

        # 尝试匹配完整类型
        color = node_type_colors.get(node_type, None)

        # 如果未匹配，尝试匹配主要部分
        if color is None:
            main_type = node_type.split('_')[0]
            color = node_type_colors.get(main_type, None)

        # 如果仍未匹配，使用默认颜色并记录类型
        if color is None:
            color = '#FFFFFF'
            unmatched_types.add(node_type)

        node_colors.append(color)

    # 如果有未匹配的类型，打印警告
    if unmatched_types:
        print(f"警告: 以下节点类型未匹配颜色: {unmatched_types}")

    nx.draw_networkx_nodes(
        G, node_positions,
        node_size=1500,
        node_color = node_colors,
        alpha=0.9,
        edgecolors='#333333',
        linewidths=1.5
    )

    # 10. 绘制边（带专业样式）
    edge_colors = {
        'AST': '#3498db',  # 蓝色
        'CFG': '#2ecc71',  # 绿色
        'CALL': '#e74c3c'  # 红色
    }

    # 按注意力权重排序边（重要边最后绘制）
    all_edges = sorted(G.edges(data=True), key=lambda x: x[2].get('attention', 0))

    for u, v, attrs in all_edges:
        edge_type = attrs.get('original_type', '')
        attention = attrs.get('attention', 0.0)

        nx.draw_networkx_edges(
            G, node_positions,
            edgelist=[(u, v)],
            width=attention * 12 + 1.5,
            edge_color=edge_colors.get(edge_type, '#CCCCCC'),
            alpha=min(0.4 + attention * 0.6, 1.0),
            arrowsize=20,
            arrowstyle='-|>',
            connectionstyle='arc3,rad=0.1'  # 添加轻微弧度
        )

    # 11. 专业节点标签绘制
    font = FontProperties()
    font.set_family('sans-serif')
    font.set_size(14)

    for node, position in node_positions.items():
        node_data = G.nodes[node]
        label = node_data['full_label']

        # 创建文本框
        text = ax.text(
            position[0], position[1],
            label,
            fontproperties=font,
            ha='center', va='center',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='white',
                edgecolor='#888888',
                alpha=0.95,
                linewidth=1.2
            )
        )

        # 添加轻微阴影增强可读性
        from matplotlib.patheffects import withStroke
        text.set_path_effects([
            withStroke(
                linewidth=2,
                foreground='white'
            )
        ])

    # 12. 边标签绘制（专业样式）
    edge_labels = {}
    for u, v in G.edges():
        attention = G.edges[u, v].get('attention', 0.0)
        if attention > 0.1:
            edge_labels[(u, v)] = f"{attention:.2f}"

    nx.draw_networkx_edge_labels(
        G,
        pos=node_positions,
        edge_labels=edge_labels,
        font_size=13,
        font_color='#8B0000',  # 深红色
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='#CCCCCC',
            alpha=0.9
        ),
        label_pos=0.5
    )

    # 13. 专业图例
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # 节点类型图例
    node_legend_elements = []
    for node_type, color in node_type_colors.items():
        node_legend_elements.append(
            Patch(facecolor=color, edgecolor='#333333', label=node_type)
        )

    # 边类型图例
    edge_legend_elements = [
        Line2D([0], [0], color='#3498db', lw=3, label='AST (Structure)'),
        Line2D([0], [0], color='#2ecc71', lw=3, label='CFG (Control Flow)'),
        Line2D([0], [0], color='#e74c3c', lw=3, label='CALL (Function Call)')
    ]

    # 创建两个图例
    node_legend = ax.legend(
        handles=node_legend_elements,
        loc='upper left',
        title="Node Types",
        frameon=True,
        framealpha=0.9,
        edgecolor='#CCCCCC'
    )

    ax.add_artist(node_legend)  # 添加第一个图例

    ax.legend(
        handles=edge_legend_elements,
        loc='upper right',
        title="Edge Types",
        frameon=True,
        framealpha=0.9,
        edgecolor='#CCCCCC'
    )

    # 14. 专业标题和注释
    graph_name = getattr(graph_data, 'graph_name', 'Unknown Graph')
    plt.title(
        f"Attention Visualization: {graph_name}",
        fontsize=20,
        pad=20,
        fontweight='bold',
        color='#333333'
    )

    plt.figtext(
        0.5, 0.02,
        "Line thickness represents attention weight | Node color indicates type",
        ha="center",
        fontsize=14,
        color='#555555'
    )

    # 15. 布局优化
    plt.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, bottom=0.05)

    # 16. 保存高质量图片
    output_path = f"professional_attention_{graph_name.replace('.json', '_new.pdf')}"
    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches='tight',
        facecolor=ax.get_facecolor()
    )
    plt.close()

    return output_path, "Professional visualization completed"
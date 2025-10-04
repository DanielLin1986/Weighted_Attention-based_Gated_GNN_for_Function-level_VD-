
from src.utils import PROJECT_ROOT
import json
from nltk.tokenize import RegexpTokenizer
import numpy as np
import os
import pickle
import re
import shutil
import subprocess
import torch
import torch_geometric.data as geom_data
import logging
from src.model.CodeBERT_model import Model
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
#from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, RobertaModel,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TorchGraphPrepError(Exception):
    """ Error in preparing data for a torch graph
    """
    pass

class JoernExeError(Exception):
    """ Error in running Joern
    """
    pass

class DatasetBuilder:
    """A class to set up the project dataset. 

    Use this to generate or reuse the dataset source files and json graphs, and
        to build the torch data.

    Attrributes:
        embedding_size: the size of the embedding for a single node in a graph
        label_encode: Dict[str, numpy.array] mapping AST labels to their embedding
        project_dir: Path of the base directory for the project
        tokenizer: RegexpTokenizer tokenize C source code
        word_vectors: gensim Word2VecKeyedVectors, encode word vectors for source code

    """
    def __init__(self, fresh_build: bool, test_build: bool) -> None:
        """Set up the builder
        
        Args:
            fresh_build: True to generate dataset source files and graphs,
                deleting any existing. False to reuse existing files.
            test_build: True to build on a sample of the dataset.
            
        Raises:
            JoernExeError: if running Joern fails.
        """
        c_regexp = r'\w+|->|\+\+|--|<=|>=|==|!=|' + \
                   r'<<|>>|&&|\|\||-=|\+=|\*=|/=|%=|' + \
                   r'&=|<<=|>>=|^=|\|=|::|' + \
                   r"""[!@#$%^&*()_+-=\[\]{};':"\|,.<>/?]"""

        self.tokenizer = RegexpTokenizer(c_regexp)
        if test_build:
            self.project_dir = os.path.join(PROJECT_ROOT, 'sample_for_test')
        else:
            self.project_dir = str(PROJECT_ROOT)
        # 检测GPU可用性
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        # CodeBERT initialization
        #config_class, model_class, tokenizer_class = MODEL_CLASSES['roborta']
        self.codbert_config = RobertaConfig.from_pretrained(r'D:\Research\CodeBERT\codebert-base', cache_dir=None, num_labels=1)
        self.codebert_tokenizer = RobertaTokenizer.from_pretrained(r'D:\Research\CodeBERT\codebert-base', do_lower_case=False, cache_dir=None)
        self.codebert_model = RobertaModel.from_pretrained(r'D:\Research\CodeBERT\codebert-base', from_tf=False, config=self.codbert_config, cache_dir=None)

        self.model1 = Model(self.codebert_model, self.codbert_config, self.codebert_tokenizer)
        # 加载微调模型的状态字典（带键名修复）
        state_dict_path = r'D:\Research\CodeBERT\fine-tuning_model.bin'
        pretrained_dict = torch.load(state_dict_path, map_location=self.device)

        # 修复键名：移除"encoder.roberta."前缀中的"roberta."
        fixed_dict = {}
        for key, value in pretrained_dict.items():
            new_key = key.replace("encoder.roberta.", "encoder.")
            fixed_dict[new_key] = value

        # 4. 加载修复后的状态字典
        self.model1.load_state_dict(fixed_dict, strict=False)  # 使用strict=False忽略不匹配的键

        self.model1.to(self.device)
        self.model1.eval()  # 设置为评估模式

        # 冻结CodeBERT参数（可选，根据内存和性能需求）
        for param in self.codebert_model.parameters():
            param.requires_grad = False

        # 编码标签
        self.encode_labels()
        model_dir = os.path.join(self.project_dir, 'models')
        label_set_path = os.path.join(model_dir, 'label_set.pickle')
        with open(label_set_path, 'rb') as label_f:
            label_set = pickle.load(label_f)
        labels = sorted(list(label_set))
        # OHE in this manner b/c doing one sample at a time
        self.label_encode = {}
        for i in range(len(labels)):
            vec = np.zeros(len(labels))
            vec[i] = 1.
            self.label_encode[labels[i]] = vec

        # 设置嵌入大小
        self.embedding_size = 768 + len(self.label_encode)  # CodeBERT隐藏层大小768 + 标签嵌入大小

    def clean_code(self, code: str) -> str:
        """Enhanced code cleaning pipeline"""
        if not code:
            return ""

        # Standard replacements
        code = (code.replace('\\n', ' ')
                .replace('\\t', ' ')
                .replace('\\"', '"'))

        # Special symbol handling
        symbol_map = {
            '{': ' { ', '}': ' } ',
            '(': ' ( ', ')': ' ) ',
            ';': ' ; ', '=': ' = ',
            '<': ' < ', '>': ' > '
        }
        for k, v in symbol_map.items():
            code = code.replace(k, v)

        # CamelCase splitting
        code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)

        code = re.sub(r'good|bad|g2b|b2g|B2G|G2B', 'str', code) #  # For SARD

        # Remove redundant spaces
        return re.sub(r'\s+', ' ', code).strip()

    def _get_codebert_embedding(self, code: str) -> torch.Tensor:
        """使用CodeBERT获取代码片段的嵌入向量"""
        if not code:
            return torch.zeros(768)

        # 清理代码
        clean_code = self.clean_code(code)

        try:
            # 编码文本
            inputs = self.codebert_tokenizer(
                clean_code,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding="max_length"
            ).to(self.device)  # 关键：移动到GPU`

            # 获取嵌入
            with torch.no_grad():
                outputs = self.codebert_model(**inputs)
                # 使用[CLS]标记作为整个序列的表示
                embeding = outputs.last_hidden_state[0, 0, :]
                return embeding.cpu()
        except Exception as e:
            logging.error(f"CodeBERT处理错误: {e}, 代码片段: '{clean_code[:50]}...'")
            return torch.zeros(768)

    def _get_node_embedding(self, node: Dict) -> torch.Tensor:
        """获取节点嵌入向量（使用CodeBERT）"""
        # 获取标签嵌入
        raw_label = node.get("_label", "UNKNOWN").strip("'\"")
        if raw_label in self.label_encode:
            label_vec = torch.tensor(self.label_encode[raw_label], dtype=torch.float)
        else:
            label_vec = torch.zeros(len(self.label_encode))

        # 确保label_vec长度为15（不足则补零）
        if label_vec.size(0) < 15:
            # 计算需要补充的零的数量
            padding_size = 15 - label_vec.size(0)
            # 在末尾补零
            label_vec = torch.cat([label_vec, torch.zeros(padding_size)])
        elif label_vec.size(0) > 15:
            # 如果长度超过15，截断到前15个元素
            label_vec = label_vec[:15]

        # 获取代码嵌入
        code = node.get('code', '')
        code_embedding = self._get_codebert_embedding(code)

        # 拼接嵌入向量
        return torch.cat([code_embedding, label_vec])

    def build_graphs(self, num_nodes: int) -> List[geom_data.Data]:
        """Get a list of torch graphs for the project's dataset

        Returns:
            A list of torch graph objects
        """
        # setup data locations
        graph_dir = os.path.join(self.project_dir, 'data', 'sard' + os.sep + 'test_graphs')
        [(_, _, graph_name_list)] = [x for x in os.walk(graph_dir)]
        #setup containers
        dataset_id = []
        dataset = []
        graph_build_err_list = []

        #setup encoders
        for f_name in graph_name_list:
            try:
                torch_graph = self.prepare_torch_data(f_name, num_nodes)
                #if 'cve' in f_name.lower():
                #  vul_torch_graph = self.prepare_torch_data_for_visualization(f_name, num_nodes)
                #  dataset.append(vul_torch_graph)
                #  dataset_id.append(f_name)
            except TorchGraphPrepError:    
                graph_build_err_list.append(f_name)
                continue
            dataset.append(torch_graph)
            dataset_id.append(f_name)
        log_path = os.path.join(self.project_dir, "log", "graph_build_err.log")
        with open(log_path, 'w') as log_f:
            for f_name in graph_build_err_list:
                log_f.write(f_name + '\n')
        return dataset, dataset_id

    def cleanup(self) -> None:
        """Clean up the project directory, removing generated files/data

        Requires:
            self.project_dir has been assigned
        """
        def clean(dir_path):
            [(_, _, files)] = [x for x in os.walk(dir_path)]
            for f in files: os.remove(os.path.join(dir_path, f))
        data_dir = os.path.join(self.project_dir, 'data')
        src_file_dir = os.path.join(data_dir, 'src_files')
        graph_dir = os.path.join(data_dir, 'graphs')
        log_dir = os.path.join(self.project_dir, 'log')
        model_dir = os.path.join(self.project_dir, 'models')
        # Find all subdirectories to the project that are the workspace generated by joern
        workspace_dir = os.path.join(self.project_dir, 'workspace')
        if os.path.isdir(workspace_dir):
            shutil.rmtree(workspace_dir)
        cleaned_dir_list = [src_file_dir, graph_dir, log_dir, model_dir]
        for dir_path in cleaned_dir_list: clean(dir_path)

    def create_src_files(self) -> None:
        """Creates source files of raw dataset. File name is <unique id>+<target label>
        """
        data_dir = os.path.join(self.project_dir, "data")
        dataset_path = os.path.join(data_dir, "total.jsonl")
        src_file_dir = os.path.join(data_dir, "src_files")
        with open(dataset_path, 'r') as dataset_f:
            dataset = json.load(dataset_f)
            i = 0 # index for unique observation key
            for i, sample in enumerate(dataset):
                file_name = f"{i}_{sample['target']}.c"
                with open(os.path.join(src_file_dir, file_name), 'w') as write_f:
                    write_f.write(sample["func"])

    def encode_labels(self) -> None:
        """Save the label encoding for AST node labels
        """
        graph_dir = os.path.join(self.project_dir, "data", 'sard' + os.sep + 'test_graphs')
        [(_, _, graph_name_list)] = [x for x in os.walk(graph_dir)]
        label_set = set({})
        for f_name in graph_name_list:
            with open(os.path.join(graph_dir, f_name), encoding='latin1') as f:
                json_data = json.load(f)
                ast_nodes = json_data['ast_nodes']
                for node in ast_nodes:
                    label_set.add(node["_label"])
        model_dir = os.path.join(self.project_dir,  "models")
        with open(os.path.join(model_dir, "label_set.pickle"), "wb") as label_f:
            pickle.dump(label_set, label_f)

    def get_target(self, file_name: str) -> int:
        """Get the target given a file name
        """
        #target_str = file_name.split("_")[1] # For real-world dataset
        target_str = file_name.split("_")[-1] # For SARD dataset
        non_num = re.compile(r'[^\d]')
        target = int(non_num.sub('', target_str))
        return target

        # Code Improved

    def find_parent_method(self, node: Dict, json_data: Dict) -> Optional[int]:
        """在AST树中向上查找包含该节点的METHOD节点"""
        # 构建AST父节点映射（子节点ID -> 父节点ID）
        parent_map = {}
        for src, dst in json_data.get('ast_edges', []):
            parent_map[dst] = src  # AST边是父->子，所以反向映射

        # 构建节点ID到节点对象的映射
        node_dict = {n['id']: n for n in json_data.get('ast_nodes', [])}

        # 从当前节点开始向上遍历AST树
        current_id = node['id']
        while current_id in parent_map:
            parent_id = parent_map[current_id]
            parent_node = node_dict.get(parent_id)

            # 找到METHOD节点则返回
            if parent_node and parent_node.get('_label') == 'METHOD':
                return parent_id

            # 继续向上遍历
            current_id = parent_id

        # 如果到达根节点仍未找到METHOD节点
        logger.warning(f"未找到节点 {node['id']} 的父方法节点")
        return None

    def extract_implicit_relations(self, json_data):
        """从现有数据中提取隐含关系(调用关系、数据流关系)"""

        # 从CALL节点提取调用关系
        call_edges = []
        for node in json_data.get('ast_nodes', []):
            if node.get('_label') == 'CALL':
                # 找到调用者（通常是包含该CALL的METHOD）
                caller_id = self.find_parent_method(node, json_data)
                if caller_id:
                    call_edges.append([caller_id, node.get('id')])

        # 从CFG边中提取数据流关系（CFG边本身就包含数据流信息）
        data_flow_edges = json_data.get('cfg_edges', [])

        return call_edges, data_flow_edges

    def prepare_torch_data(self, file_name: str, num_nodes: Optional[int] = None) -> geom_data.Data:
        """Optimized pipeline with edge type features"""
        # 1. Fast JSON loading
        graph_path = os.path.join(self.project_dir, 'data', 'sard' + os.sep + 'test_graphs', file_name)
        with open(graph_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 2. Node processing (CodeBERT)
            node_embeddings = {}
            for node in json_data['ast_nodes'][:num_nodes]:
                node_id = node['id']
                try:
                    embedding = self._get_node_embedding(node)
                    node_embeddings[node_id] = embedding
                except Exception as e:
                    logging.error(f"Embedding Error： (ID {node_id}): {e}")
                    # 使用零向量作为回退
                    node_embeddings[node_id] = torch.zeros(self.embedding_size)

        # 3. Create ID mapping (no truncation)
        old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(node_embeddings.keys())}

        # 4. Edge processing with type features
        edge_indices = []
        edge_types = []

        # AST edges (type 0)c
        for src, dst in json_data['ast_edges']:
            if src in old_to_new_id and dst in old_to_new_id:
                edge_indices.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(0)  # AST edge type

        # CFG edges (type 1)
        for src, dst in json_data['cfg_edges']:
            if src in old_to_new_id and dst in old_to_new_id:
                edge_indices.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(1)  # CFG edge type

        call_edges, additional_data_flow = self.extract_implicit_relations(json_data)

        # 添加调用边 (2)
        for src, dst in call_edges:
            if src in old_to_new_id and dst in old_to_new_id:
                edge_indices.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(2)

        # 5. Convert to tensors
        x = torch.stack([torch.FloatTensor(node_embeddings[old_id])
                         for old_id in old_to_new_id.keys()])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)

        return geom_data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([self.get_target(file_name)], dtype=torch.long)
        )

    def prepare_torch_data_for_visualization(self, file_name: str, num_nodes: Optional[int] = None) -> geom_data.Data:
        """优化的数据准备流程，保存原始信息用于注意力可视化"""
        # 1. 加载JSON数据
        graph_path = os.path.join(self.project_dir, 'data', 'sard', 'test_graphs', file_name)
        with open(graph_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 2. 提取原始节点信息（用于后续可视化）
        raw_nodes = []
        for node in json_data['ast_nodes'][:num_nodes]:
            # 保存关键属性用于可视化
            raw_nodes.append({
                'id': node.get('id'),
                'label': node.get('_label', 'UNKNOWN'),
                'code': node.get('code', ''),
                'line_number': node.get('lineNumber', -1),
                'column_number': node.get('columnNumber', -1)
            })

        # 3. 节点嵌入处理
        node_embeddings = {}
        for node in json_data['ast_nodes'][:num_nodes]:
            node_id = node['id']
            try:
                embedding = self._get_node_embedding(node)
                node_embeddings[node_id] = embedding
            except Exception as e:
                logging.error(f"节点嵌入错误 (ID {node_id}): {e}")
                # 使用零向量作为回退
                node_embeddings[node_id] = torch.zeros(self.embedding_size)

        # 4. ID映射
        original_ids = list(node_embeddings.keys())  # 保存原始ID列表
        old_to_new_id = {old_id: new_id for new_id, old_id in enumerate(original_ids)}

        # 5. 边处理（保存原始边信息）
        edges = []
        edge_types = []
        raw_edges = []  # 保存原始边信息用于可视化

        # AST边 (类型0)
        for src, dst in json_data['ast_edges']:
            if src in old_to_new_id and dst in old_to_new_id:
                edges.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(0)
                raw_edges.append({
                    'src': src,
                    'dst': dst,
                    'type': 0,
                    'original_type': 'AST'
                })

        # CFG边 (类型1)
        for src, dst in json_data['cfg_edges']:
            if src in old_to_new_id and dst in old_to_new_id:
                edges.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(1)
                raw_edges.append({
                    'src': src,
                    'dst': dst,
                    'type': 1,
                    'original_type': 'CFG'
                })

        # 提取调用关系
        call_edges, additional_data_flow = self.extract_implicit_relations(json_data)

        # 添加调用边 (类型2)
        for src, dst in call_edges:
            if src in old_to_new_id and dst in old_to_new_id:
                edges.append([old_to_new_id[src], old_to_new_id[dst]])
                edge_types.append(2)
                raw_edges.append({
                    'src': src,
                    'dst': dst,
                    'type': 2,
                    'original_type': 'CALL'
                })

        # 6. 转换为张量
        x = torch.stack([torch.FloatTensor(node_embeddings[old_id])
                         for old_id in original_ids])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_types, dtype=torch.long).unsqueeze(1)

        # 7. 创建PyG数据对象，添加原始信息属性
        data = geom_data.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([self.get_target(file_name)], dtype=torch.long)
        )

        # 添加用于注意力可视化的元数据
        data.raw_nodes = raw_nodes  # 原始节点信息
        data.original_ids = original_ids  # 原始ID列表
        data.raw_edges = raw_edges  # 原始边信息

        # 添加文件标识
        data.graph_name = file_name

        return data
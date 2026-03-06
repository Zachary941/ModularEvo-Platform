"""
步骤4：混合数据集构造与 DataLoader。

负责加载 4 个任务的原始数据，创建标签映射，按 Dirichlet 随机比例混合采样，
输出统一格式的 Batch：{input_ids, attention_mask, labels, task_ids}。
"""

import os
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

from .config import PROJECT_ROOT, TASK_NAMES, TASK_CONFIGS, TASK_NAME_TO_IDX

logger = logging.getLogger(__name__)

# ===== 数据路径配置 =====
DATA_PATHS = {
    'code': {
        'type': 'parquet_single',  # 单个 parquet（全量 79k，包含全部 1006 种语言）
        'path': os.path.join(PROJECT_ROOT, 'fintune/data/code/train-00000-of-00001-8b4da49264116bbf.parquet'),
        'text_field': 'code',
        'label_field': 'language_name',
        'label_type': 'string',  # 需要 sorted→映射
        'train_ratio': 0.6,   # 与微调代码一致：60% train
        'val_ratio': 0.1,     # 10% validation
        'test_ratio': 0.3,    # 30% test
        'seed': 42,
    },
    'langid': {
        'type': 'csv_custom',  # 自定义的 Nordic DSL 格式
        'train_path': os.path.join(PROJECT_ROOT, 'fintune/data/langid/nordic_dsl_10000train.csv'),
        'test_path': os.path.join(PROJECT_ROOT, 'fintune/data/langid/nordic_dsl_10000test.csv'),
        'text_field': 'sentence',
        'label_field': 'language',
        'label_type': 'string',
    },
    'law': {
        'type': 'parquet',
        'dir': os.path.join(PROJECT_ROOT, 'fintune/data/lex_glue/scotus'),
        'text_field': 'text',
        'label_field': 'label',
        'label_type': 'int',  # 已经是整数
    },
    'math': {
        'type': 'parquet',
        'dir': os.path.join(PROJECT_ROOT, 'fintune/data/mathqa'),
        'text_field': 'question',
        'label_field': 'topic',
        'label_type': 'string',
    },
}

# ===== 分词配置 =====
MAX_LENGTH = 512
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'data', 'gpt-neo-125m')


# ===== LangID CSV 解析 =====
def _load_langid_csv(filepath):
    """
    解析 Nordic DSL 格式：每行为 "前缀, 文本内容 语言代码"
    返回 list[dict] with keys: sentence, language
    """
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 去除 "dataset10000, " 或 "dataset50000, " 前缀
            if line.startswith('dataset'):
                comma_idx = line.index(',')
                line = line[comma_idx + 1:].strip()
            # 最后2个字符是语言代码
            language = line[-2:]
            sentence = line[:-2].strip()
            samples.append({'sentence': sentence, 'language': language})
    return samples


# ===== 标签映射创建 =====
def _create_label_mapping(values):
    """从字符串标签列表创建 sorted→index 映射（与微调代码一致）。"""
    unique = sorted(set(values))
    return {v: i for i, v in enumerate(unique)}


# ===== 单任务数据集 =====
class TaskDataset(Dataset):
    """
    单任务数据集：存储 tokenize 后的 input_ids, attention_mask, label。
    """
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.labels = labels
        self.encodings = tokenizer(
            texts,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': self.labels[idx],
        }


# ===== 数据加载器 =====
def load_tokenizer():
    """加载 GPT2 Tokenizer。"""
    tokenizer = GPT2Tokenizer.from_pretrained(TOKENIZER_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_task_datasets(split='train', max_samples_per_task=None):
    """
    加载所有任务的数据集。

    Args:
        split: 'train', 'validation' 或 'test'
        max_samples_per_task: 每个任务最大样本数（None=全部）

    Returns:
        task_datasets: Dict[str, TaskDataset]
        label_mappings: Dict[str, dict] — 各任务的标签映射
    """
    tokenizer = load_tokenizer()
    task_datasets = {}
    label_mappings = {}

    for task_name in TASK_NAMES:
        cfg = DATA_PATHS[task_name]
        logger.info(f"加载 {task_name} ({split}) 数据...")

        texts, raw_labels = _load_raw_data(task_name, cfg, split)

        # 创建标签映射
        if cfg['label_type'] == 'string':
            # 需要建立完整映射（包括所有 split 的标签以保持一致）
            # 使用当前数据的标签构建映射
            all_labels = _collect_all_labels(task_name, cfg)
            label_map = _create_label_mapping(all_labels)
            label_mappings[task_name] = label_map
            labels = [label_map[l] for l in raw_labels]
        else:
            # 已经是整数
            label_mappings[task_name] = None
            labels = [int(l) for l in raw_labels]

        # 限制样本数
        if max_samples_per_task and len(texts) > max_samples_per_task:
            indices = np.random.choice(len(texts), max_samples_per_task, replace=False)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]

        logger.info(f"  {task_name}: {len(texts)} 样本, "
                     f"num_classes={TASK_CONFIGS[task_name]['num_classes']}")

        # 创建 TokenizedDataset
        task_datasets[task_name] = TaskDataset(texts, labels, tokenizer)

    return task_datasets, label_mappings


def _load_raw_data(task_name, cfg, split):
    """加载原始数据（未 tokenize），返回 (texts, labels)。"""
    text_field = cfg['text_field']
    label_field = cfg['label_field']

    if cfg['type'] == 'parquet_single':
        # Code 数据集：从全量 parquet 加载，手动划分 train/val/test
        import pandas as pd
        df = pd.read_parquet(cfg['path'])
        # 与微调代码一致的划分方式：shuffle + split
        df = df.sample(frac=1, random_state=cfg.get('seed', 42)).reset_index(drop=True)
        n = len(df)
        n_test = int(n * cfg.get('test_ratio', 0.3))
        n_val = int(n * cfg.get('val_ratio', 0.1))
        if split == 'test':
            df_split = df[:n_test]
        elif split == 'validation':
            df_split = df[n_test:n_test + n_val]
        else:  # train
            df_split = df[n_test + n_val:]
        texts = df_split[text_field].tolist()
        raw_labels = df_split[label_field].tolist()

    elif cfg['type'] == 'csv_custom':
        # LangID：自定义 CSV 解析
        if split == 'test':
            samples = _load_langid_csv(cfg['test_path'])
        else:
            samples = _load_langid_csv(cfg['train_path'])
            if split == 'validation':
                # 取最后 10% 作为验证集（与微调代码一致）
                n_val = max(1, len(samples) // 10)
                samples = samples[-n_val:]
            elif split == 'train':
                n_val = max(1, len(samples) // 10)
                samples = samples[:-n_val]
        texts = [s[text_field] for s in samples]
        raw_labels = [s[label_field] for s in samples]

    elif cfg['type'] == 'parquet':
        data_dir = cfg['dir']
        # 确定文件名
        if split == 'validation' and task_name == 'math':
            fname = 'val-00000-of-00001.parquet'  # math 用 val- 前缀
        elif split == 'validation':
            fname = 'validation-00000-of-00001.parquet'
        elif split == 'test':
            fname = 'test-00000-of-00001.parquet'
        else:
            fname = 'train-00000-of-00001.parquet'

        filepath = os.path.join(data_dir, fname)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"数据文件不存在: {filepath}")

        import pandas as pd
        df = pd.read_parquet(filepath)
        texts = df[text_field].tolist()
        raw_labels = df[label_field].tolist()

    else:
        raise ValueError(f"未知数据类型: {cfg['type']}")

    return texts, raw_labels


def _collect_all_labels(task_name, cfg):
    """收集一个任务所有 split 的标签（用于建立完整的 sorted 映射）。"""
    all_labels = []

    if cfg['type'] == 'parquet_single':
        import pandas as pd
        df = pd.read_parquet(cfg['path'])
        all_labels.extend(df[cfg['label_field']].tolist())

    elif cfg['type'] == 'csv_custom':
        for path in [cfg['train_path'], cfg['test_path']]:
            samples = _load_langid_csv(path)
            all_labels.extend([s[cfg['label_field']] for s in samples])

    elif cfg['type'] == 'parquet':
        import pandas as pd
        data_dir = cfg['dir']
        for fname in os.listdir(data_dir):
            if fname.endswith('.parquet'):
                df = pd.read_parquet(os.path.join(data_dir, fname))
                if cfg['label_field'] in df.columns:
                    all_labels.extend(df[cfg['label_field']].tolist())

    return all_labels


# ===== Dirichlet 混合采样 =====
class MixedBatchSampler:
    """
    按 Dirichlet 随机比例从各任务数据集中采样混合 Batch。
    每个 Batch 的任务比例随机生成，训练 Router 感知不同分布。
    """

    def __init__(self, task_datasets, batch_size=32, num_batches=5000,
                 dirichlet_alpha=1.0, seed=None, pure_task_prob=0.0):
        """
        Args:
            task_datasets: Dict[str, TaskDataset]
            batch_size: 每个 Batch 的总样本数
            num_batches: 生成的 Batch 总数（一个 epoch）
            dirichlet_alpha: Dirichlet 分布的浓度参数
                             α=1.0 均匀, α<1.0 偏向极端, α>1.0 偏向均匀
            seed: 随机种子（可选）
            pure_task_prob: 生成纯单任务 batch 的概率（如 0.25 表示 25%）
        """
        self.task_datasets = task_datasets
        self.task_names = list(task_datasets.keys())
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.dirichlet_alpha = dirichlet_alpha
        self.pure_task_prob = pure_task_prob
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for _ in range(self.num_batches):
            yield self._sample_one_batch()

    def _sample_one_batch(self):
        """采样一个混合 Batch（一定概率生成纯单任务 Batch）。"""
        # 方案一：以 pure_task_prob 的概率生成纯单任务 batch
        if self.pure_task_prob > 0 and self.rng.random() < self.pure_task_prob:
            # 随机选择一个任务，整个 batch 全部来自该任务
            chosen_idx = self.rng.randint(len(self.task_names))
            ratios = np.zeros(len(self.task_names))
            ratios[chosen_idx] = 1.0
        else:
            # 1. Dirichlet 随机生成混合比例
            ratios = self.rng.dirichlet(
                np.ones(len(self.task_names)) * self.dirichlet_alpha
            )

        # 2. 按比例分配样本数（确保总数 = batch_size）
        counts = (ratios * self.batch_size).astype(int)
        # 处理余数：分配给比例最大的任务
        remainder = self.batch_size - counts.sum()
        if remainder > 0:
            top_idx = np.argsort(ratios)[-remainder:]
            counts[top_idx] += 1

        # 3. 从各任务中随机无放回采样（如果样本不足则有放回）
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_task_ids = []

        for i, task_name in enumerate(self.task_names):
            count = int(counts[i])
            if count == 0:
                continue

            ds = self.task_datasets[task_name]
            n = len(ds)
            replace = count > n
            indices = self.rng.choice(n, count, replace=replace)

            for idx in indices:
                sample = ds[idx]
                all_input_ids.append(sample['input_ids'])
                all_attention_mask.append(sample['attention_mask'])
                all_labels.append(sample['label'])
                all_task_ids.append(task_name)

        # 4. Shuffle batch 内的顺序
        perm = self.rng.permutation(len(all_input_ids))
        all_input_ids = [all_input_ids[p] for p in perm]
        all_attention_mask = [all_attention_mask[p] for p in perm]
        all_labels = [all_labels[p] for p in perm]
        all_task_ids = [all_task_ids[p] for p in perm]

        # 5. 组装 Batch
        batch = {
            'input_ids': torch.stack(all_input_ids),           # [B, Seq]
            'attention_mask': torch.stack(all_attention_mask),  # [B, Seq]
            'labels': torch.tensor(all_labels, dtype=torch.long),  # [B]
            'task_ids': all_task_ids,                           # List[str], len=B
        }
        return batch


def create_mixed_dataloader(split='train', batch_size=32, num_batches=5000,
                            dirichlet_alpha=1.0, max_samples_per_task=None,
                            seed=None):
    """
    一站式创建混合 DataLoader。

    Args:
        split: 数据分割 ('train', 'validation', 'test')
        batch_size: 每 Batch 样本数
        num_batches: 总 Batch 数
        dirichlet_alpha: Dirichlet α 参数
        max_samples_per_task: 每任务最大样本数
        seed: 随机种子

    Returns:
        sampler: MixedBatchSampler (可直接 iterate)
        task_datasets: Dict[str, TaskDataset]
        label_mappings: Dict[str, dict]
    """
    task_datasets, label_mappings = load_task_datasets(
        split=split, max_samples_per_task=max_samples_per_task
    )

    sampler = MixedBatchSampler(
        task_datasets=task_datasets,
        batch_size=batch_size,
        num_batches=num_batches,
        dirichlet_alpha=dirichlet_alpha,
        seed=seed,
    )

    return sampler, task_datasets, label_mappings


def create_fixed_count_batches(task_datasets, total_task_counts, batch_size, seed=None):
    """
    按固定任务样本数构造混合 batch。

    该函数主要用于验证/评测：给定一组目标任务配比，按比例拆成多个 batch，
    并在数据量不足时循环复用小数据集样本，保证不同阶段使用同一套混合构造逻辑。

    Args:
        task_datasets: Dict[str, TaskDataset]
        total_task_counts: Dict[str, int]，各任务的目标总样本数
        batch_size: 每个 batch 的样本数
        seed: 随机种子（可选）

    Returns:
        List[dict]，每个元素格式与 MixedBatchSampler 产出的 batch 一致
    """
    total = sum(int(total_task_counts.get(task_name, 0)) for task_name in TASK_NAMES)
    if total <= 0:
        return []

    rng = np.random.RandomState(seed)
    num_batches = max(1, total // batch_size)

    ratios = {
        task_name: int(total_task_counts.get(task_name, 0)) / total
        for task_name in TASK_NAMES
    }
    per_batch_counts = {}
    remaining = batch_size
    for i, task_name in enumerate(TASK_NAMES):
        if i == len(TASK_NAMES) - 1:
            per_batch_counts[task_name] = remaining
        else:
            if total_task_counts.get(task_name, 0) == 0:
                per_batch_counts[task_name] = 0
            else:
                count = max(1, int(round(batch_size * ratios[task_name])))
                count = min(count, remaining)
                per_batch_counts[task_name] = count
            remaining -= per_batch_counts[task_name]

    task_indices = {}
    for task_name in TASK_NAMES:
        ds = task_datasets[task_name]
        shuffled = rng.permutation(len(ds)).tolist()
        task_indices[task_name] = {
            'indices': shuffled,
            'pos': 0,
            'size': len(ds),
        }

    batches = []
    for _ in range(num_batches):
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        all_task_ids = []

        for task_name in TASK_NAMES:
            count = per_batch_counts[task_name]
            if count == 0:
                continue

            ds = task_datasets[task_name]
            info = task_indices[task_name]

            for _ in range(count):
                idx = info['indices'][info['pos'] % info['size']]
                info['pos'] += 1
                sample = ds[idx]
                all_input_ids.append(sample['input_ids'])
                all_attention_mask.append(sample['attention_mask'])
                all_labels.append(sample['label'])
                all_task_ids.append(task_name)

        batch = {
            'input_ids': torch.stack(all_input_ids),
            'attention_mask': torch.stack(all_attention_mask),
            'labels': torch.tensor(all_labels, dtype=torch.long),
            'task_ids': all_task_ids,
        }
        batches.append(batch)

    return batches


# ===== 评估用：单任务 DataLoader =====
def create_single_task_dataloader(task_name, split='test', batch_size=32,
                                  max_samples=None):
    """
    创建单任务 DataLoader（用于评估阶段）。

    Returns:
        dataloader: DataLoader
        label_mapping: dict or None
    """
    tokenizer = load_tokenizer()
    cfg = DATA_PATHS[task_name]

    texts, raw_labels = _load_raw_data(task_name, cfg, split)

    if cfg['label_type'] == 'string':
        all_labels = _collect_all_labels(task_name, cfg)
        label_map = _create_label_mapping(all_labels)
        labels = [label_map[l] for l in raw_labels]
    else:
        label_map = None
        labels = [int(l) for l in raw_labels]

    if max_samples and len(texts) > max_samples:
        indices = np.random.choice(len(texts), max_samples, replace=False)
        texts = [texts[i] for i in indices]
        labels = [labels[i] for i in indices]

    dataset = TaskDataset(texts, labels, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, label_map


# ===== 主函数：数据统计 =====
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

    logger.info("=" * 60)
    logger.info("步骤4：数据加载测试")
    logger.info("=" * 60)

    # 加载训练集
    task_datasets, label_mappings = load_task_datasets(
        split='train', max_samples_per_task=500  # 小规模测试
    )

    for name, ds in task_datasets.items():
        logger.info(f"  {name}: {len(ds)} 样本")

    # 测试混合采样
    logger.info("\n测试 Dirichlet 混合采样 (3 batches)...")
    sampler = MixedBatchSampler(
        task_datasets, batch_size=16, num_batches=3, seed=42
    )
    for i, batch in enumerate(sampler):
        task_counts = {}
        for tid in batch['task_ids']:
            task_counts[tid] = task_counts.get(tid, 0) + 1
        logger.info(f"  Batch {i}: 分布={task_counts}, "
                     f"input_ids={list(batch['input_ids'].shape)}, "
                     f"labels range=[{batch['labels'].min()}, {batch['labels'].max()}]")

    logger.info("\n✓ 数据加载测试完成！")

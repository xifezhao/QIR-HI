import numpy as np
import ir_datasets
import nltk
import string
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
import abc

# 还需要添加这些NLTK相关的导入
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# =======================================================================
# NLTK 数据下载
# =======================================================================
def download_nltk_data():
    try:
        stopwords.words('english')
    except LookupError:
        print("Downloading NLTK 'stopwords' data...")
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' data...")
        nltk.download('punkt', quiet=True)

download_nltk_data()
# =======================================================================


# --- 4. Experimental Setup (基本函数保持不变) ---

def preprocess(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return [stemmer.stem(w) for w in tokens if w.isalpha() and w not in stop_words]

def precision_at_k(y_true, y_pred, k=10):
    y_pred_k = y_pred[:k]
    relevant_preds = [doc_id for doc_id in y_pred_k if doc_id in y_true]
    return len(relevant_preds) / k

def average_precision(y_true, y_pred):
    relevant_preds = [doc_id for doc_id in y_pred if doc_id in y_true]
    if not relevant_preds: return 0.0
    score, num_hits = 0.0, 0.0
    for i, p in enumerate(y_pred):
        if p in relevant_preds:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / len(relevant_preds)

# --- 基线模型 (VSM, BM25, S-BERT) 保持不变 ---
class VSM_Ranker:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(preprocessor=lambda x: x, tokenizer=lambda x: x)
        self.doc_term_matrix, self.doc_ids = None, []
    def fit(self, processed_corpus):
        self.doc_ids = list(processed_corpus.keys())
        self.doc_term_matrix = self.vectorizer.fit_transform(processed_corpus.values())
    def rank(self, processed_query):
        query_vec = self.vectorizer.transform([processed_query])
        sim = cosine_similarity(query_vec, self.doc_term_matrix).flatten()
        return [self.doc_ids[i] for i in np.argsort(-sim)]

class BM25_Ranker:
    def __init__(self): self.bm25, self.doc_ids = None, []
    def fit(self, processed_corpus):
        self.doc_ids = list(processed_corpus.keys())
        self.bm25 = BM25Okapi(list(processed_corpus.values()))
    def rank(self, processed_query):
        scores = self.bm25.get_scores(processed_query)
        return [self.doc_ids[i] for i in np.argsort(scores)[::-1]]

class SBert_Ranker:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings, self.doc_ids = None, []
    def fit(self, raw_corpus):
        self.doc_ids = list(raw_corpus.keys())
        print("Encoding corpus for S-BERT...")
        self.doc_embeddings = self.model.encode(list(raw_corpus.values()), convert_to_tensor=True)
    def rank(self, raw_query):
        query_embedding = self.model.encode(raw_query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.doc_embeddings)[0].cpu().numpy()
        return [self.doc_ids[i] for i in np.argsort(scores)[::-1]]


# =======================================================================
# 策略一：新的“整体状态”表示法 Ranker
# =======================================================================
class QIR_SI_HolisticRanker:
    """
    QIR-SI 改进版：将整个文档/查询映射为一个整体复数向量。
    """
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        # 确保维度是偶数
        embedding_dim = self.model.get_sentence_embedding_dimension()
        if embedding_dim % 2 != 0:
            raise ValueError("Embedding dimension must be even to be split into real and imaginary parts.")
        self.complex_dim = embedding_dim // 2
        self.doc_complex_vectors = {} # 存储预计算的文档复数向量
        self.doc_ids = []

    def _get_holistic_complex_vector(self, text):
        """将单个文本编码为一个整体复数向量"""
        # 1. 获取S-BERT的实数向量表示
        embedding = self.model.encode(text)
        
        # 2. 拆分为实部和虚部
        real_part = embedding[:self.complex_dim]
        imag_part = embedding[self.complex_dim:]
        
        # 3. 构造复数向量
        complex_vec = real_part + 1j * imag_part
        
        # 可选：可以对向量进行归一化，也可以不归一化。
        # 不归一化时，向量的模长也携带了信息。
        # norm = np.linalg.norm(complex_vec)
        # if norm > 0:
        #     return complex_vec / norm
        return complex_vec

    def fit(self, raw_corpus):
        """对整个语料库进行编码并存储为复数向量"""
        print("Fitting Holistic QIR-SI Ranker (Encoding corpus to complex vectors)...")
        self.doc_ids = list(raw_corpus.keys())
        # 批量编码以提高效率
        all_texts = list(raw_corpus.values())
        all_embeddings = self.model.encode(all_texts)
        
        for i, doc_id in enumerate(self.doc_ids):
            embedding = all_embeddings[i]
            real_part = embedding[:self.complex_dim]
            imag_part = embedding[self.complex_dim:]
            self.doc_complex_vectors[doc_id] = real_part + 1j * imag_part

    def rank(self, raw_query):
        """为给定查询对所有文档进行排序"""
        scores = {}
        # 1. 获取查询的整体复数向量
        q_vec = self._get_holistic_complex_vector(raw_query)
        
        # 2. 遍历所有预计算的文档向量
        for doc_id, d_vec in self.doc_complex_vectors.items():
            # 3. 计算分数：Score = |<q|d>|^2
            # np.vdot(q, d) 计算 q的共轭转置 * d，正是内积 <q|d>
            score = np.abs(np.vdot(q_vec, d_vec))**2
            scores[doc_id] = score
        
        # 4. 排序并返回结果
        return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


# --- 主实验流程 ---
def run_experiment(dataset_name="cranfield"):
    print(f"--- Running experiment on {dataset_name} dataset ---")
    dataset = ir_datasets.load(dataset_name)
    raw_corpus = {doc.doc_id: doc.text for doc in dataset.docs_iter()}
    print("Preprocessing corpus (for classical models)...")
    processed_corpus = {doc_id: preprocess(text) for doc_id, text in raw_corpus.items()}
    
    # 初始化所有模型，包括新的改进版QIR-SI
    print("Initializing models...")
    vsm = VSM_Ranker()
    bm25 = BM25_Ranker()
    sbert = SBert_Ranker()
    # 使用新的Holistic Ranker
    qir_si_holistic = QIR_SI_HolisticRanker()

    models = {
        "VSM (TF-IDF)": vsm,
        "BM25": bm25,
        "S-BERT (Cosine Sim)": sbert, # 重命名以示区分
        "QIR-SI (Holistic)": qir_si_holistic
    }

    # 训练模型
    print("Fitting models...")
    for name, model in models.items():
        print(f"--- Fitting {name} ---")
        # Holistic和S-BERT模型需要原始文本
        if isinstance(model, (SBert_Ranker, QIR_SI_HolisticRanker)):
            model.fit(raw_corpus)
        # VSM和BM25需要预处理过的文本
        else:
            model.fit(processed_corpus)

    print("Loading relevance judgments (qrels)...")
    all_qrels = defaultdict(set)
    for qrel in dataset.qrels_iter():
        if qrel.relevance > 0: all_qrels[qrel.query_id].add(qrel.doc_id)

    # 评估
    print("Evaluating models...")
    results = {name: {'P@10': [], 'MAP': []} for name in models.keys()}

    for query in dataset.queries_iter():
        true_relevant_docs = all_qrels.get(query.query_id, set())
        if not true_relevant_docs: continue
        
        raw_query = query.text
        processed_query = preprocess(raw_query)

        for name, model in models.items():
            rank_func = model.rank
            # 根据模型需要传递不同类型的查询
            if isinstance(model, (SBert_Ranker, QIR_SI_HolisticRanker)):
                ranked_docs = rank_func(raw_query)
            else:
                ranked_docs = rank_func(processed_query)
            
            results[name]['P@10'].append(precision_at_k(true_relevant_docs, ranked_docs, k=10))
            results[name]['MAP'].append(average_precision(true_relevant_docs, ranked_docs))

    # 5.1. Overall Performance
    print("\n--- 5.1 Overall Performance Table ---")
    print(f"{'Model':<30} | {'P@10':<10} | {'MAP':<10}")
    print("-" * 55)
    for name, metrics in results.items():
        avg_p10 = np.mean(metrics['P@10']) if metrics['P@10'] else 0
        avg_map = np.mean(metrics['MAP']) if metrics['MAP'] else 0
        print(f"{'Model':<30} | {avg_p10:<10.4f} | {avg_map:<10.4f}")

if __name__ == '__main__':
    run_experiment()
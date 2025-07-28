# QIR-SI (Holistic): Quantum-Inspired Information Retrieval via Holistic Interference

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Paper**: [Quantum-Inspired Information Retrieval: Relevance as a Process of Holistic Interference](https://arxiv.org/abs/YOUR_ARXIV_ID_HERE) *(Please replace this with your paper's link, e.g., to its arXiv page)*

This repository contains the official Python implementation for the paper **"Quantum-Inspired Information Retrieval: Relevance as a Process of Holistic Interference"**.

This work challenges the dominant geometric paradigm of cosine similarity in modern information retrieval. We propose and validate a competitive alternative based on the formalism of quantum mechanics. Our model, `QIR-SI (Holistic)`, encodes the semantics of an entire text into a single, indivisible quantum state and leverages the Born rule to compute relevance. Experiments show that this model not only significantly outperforms classical sparse retrieval baselines like BM25 but also achieves performance on par with a strong S-BERT dense retrieval baseline.

## Core Ideas & Contributions

As detailed in the paper, our core contributions are:

*   **Models Relevance as Interference**: Instead of geometric similarity, we model relevance as a process of probabilistic interference via the Born rule: `|⟨q|d⟩|²`.
*   **Diagnoses "Coherence Collapse"**: We identify and name a key failure mode in term-based quantum models—"Coherence Collapse"—where incoherent term-wise summation destroys the global relevance signal.
*   **Designs a Holistic Model (QIR-SI)**: To solve this, we design a "Holistic" model that represents an entire text as a **single** quantum state derived from dense embeddings, thereby preserving semantic coherence.
*   **Validates the Paradigm's Competitiveness**: On the Cranfield benchmark, our holistic interference model performs on par with an S-BERT baseline using cosine similarity (MAP: 0.3219 vs. 0.3232), validating the quantum formalism as a viable and powerful alternative in modern dense retrieval.

## How It Works

Traditional quantum-inspired models often fail due to "Coherence Collapse." Our `QIR-SI (Holistic)` model circumvents this problem with a top-down approach:

1.  **Semantic Encoding**: A pre-trained sentence encoder (e.g., S-BERT's `all-MiniLM-L6-v2`) maps an entire text (query or document) to a high-dimensional real-valued vector `v_real ∈ ℝ^2k`. This vector captures the holistic meaning of the text.

2.  **Complex Vector Construction**: We transform this `2k`-dimensional real vector `v_real` into a `k`-dimensional complex vector (i.e., a quantum state `|ψ⟩`) by partitioning it.
    *   The first `k` elements form the **real components** of the complex vector.
    *   The latter `k` elements form the **imaginary components**.

    This mapping, `M: ℝ^2k → ℂ^k`, is defined as:
    `|ψ⟩ = M(v_real) = Σ (v_j + i ⋅ v_{j+k}) |j⟩` (for j from 1 to k)

3.  **Relevance Calculation**: The relevance score between a query `q` and a document `d` is the squared modulus of the inner product of their respective quantum states, `|q_holistic⟩` and `|d_holistic⟩`. This is a direct application of the Born rule.

    `Score(q, d) = |⟨q_holistic|d_holistic⟩|²`

    This formulation fundamentally eliminates the summation over independent terms, thus avoiding Coherence Collapse. Interference now occurs between the internal dimensions of the two holistic semantic vectors.

## Getting Started

### Requirements
*   Python 3.8+
*   PyTorch (a dependency of sentence-transformers)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/QIR-SI-Holistic.git
    cd QIR-SI-Holistic
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**
    Create a `requirements.txt` file with the following content:
    ```txt
    numpy
    ir_datasets
    nltk
    scikit-learn
    rank_bm25
    sentence-transformers
    ```
    Then, install the packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Experiment

Simply execute the main script to run the full experimental pipeline on the Cranfield dataset. The script will automatically download the necessary NLTK data and the pre-trained S-BERT model.

python run_experiment.py
The program will load the data, fit all models, run the evaluation, and print a final performance comparison table to the console.

## Experimental Results

The key results from our implementation on the Cranfield dataset, which replicate Table 5 from our paper, are shown below.

| Model                  | P@10   | MAP    |
| ---------------------- | ------ | ------ |
| **Classical Baselines**|        |        |
| VSM (TF-IDF)           | 0.2182 | 0.2709 |
| BM25                   | 0.2196 | 0.2837 |
| **Dense Baselines**    |        |        |
| S-BERT (Cosine Sim)    | 0.2440 | 0.3232 |
| **Proposed Model**     |        |        |
| **QIR-SI (Holistic)**  | **0.2444** | **0.3219** |

The results clearly demonstrate that the `QIR-SI (Holistic)` model significantly outperforms classical baselines and is fully competitive with the strong S-BERT baseline, validating the effectiveness of our approach.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

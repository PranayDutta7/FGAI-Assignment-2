# Word-Level Next-Word Prediction and Text Generation Using RNN Architectures

**Course:** Foundations of Generative AI (ICT4403)  
**Assignment:** Assignment 2  
**Author:** Pranay Dutta  
**GitHub Repository:** https://github.com/PranayDutta7/FGAI-Assignment-2

## 1. Project Overview

This project implements and compares four recurrent neural network architectures for word-level next-word prediction and autoregressive text generation:

- Simple Recurrent Neural Network (RNN)
- Bidirectional Recurrent Neural Network (BiRNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

The models are trained on the Tiny Shakespeare dataset. Given a sequence of preceding words, each model predicts the next word. Text generation is performed by repeatedly predicting and sampling the next word, appending it to the sequence, and using the updated sequence to generate subsequent words.

The project evaluates the architectures using validation loss, perplexity, next-word prediction accuracy, training time, and qualitative analysis of generated text.

## 2. Objectives

- Prepare a text corpus for word-level language modelling.
- Implement and train four recurrent neural network architectures.
- Evaluate next-word prediction using quantitative metrics.
- Generate text from a common seed using different sampling temperatures.
- Compare model performance, computational cost, diversity, and coherence.
- Visualise training behaviour and model comparison results.

## 3. Dataset

**Dataset:** Tiny Shakespeare

**Source:** https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

The dataset contains Shakespearean dialogue and literary text. It provides varied vocabulary, character names, punctuation, and dialogue structures for experimenting with language modelling.

### Preprocessing

The preprocessing pipeline:

1. Tokenises the text at the word level while retaining punctuation as separate tokens.
2. Splits the token sequence chronologically into training and validation sets.
3. Builds the vocabulary using only the training portion.
4. Maps tokens to integer IDs and represents unseen validation tokens using `<UNK>`.
5. Creates fixed-length input sequences and next-word targets.

The training/validation split is 90%/10%.

| Dataset statistic | Value |
|---|---:|
| Total tokens | 253,593 |
| Training tokens | 228,233 |
| Validation tokens | 25,360 |
| Vocabulary size | 13,486 |
| Validation unknown-token fraction | 5.07% |

The word-level processed files used by the current models are stored in `data/processed_word/`.

## 4. Model Architectures

### Simple RNN

Processes the input sequence through recurrent hidden states. It is computationally lightweight, but can struggle to retain information across long sequences.

### Bidirectional RNN (BiRNN)

Processes the supplied context in both forward and backward directions. The two final directional representations are combined to predict the next word.

The model only receives preceding context; it does not receive future target words. Bidirectionality here refers to processing the known input window in both directions.

### Long Short-Term Memory (LSTM)

Uses a cell state and gating mechanisms to regulate the flow of information. This design helps the model retain information across longer sequences.

### Gated Recurrent Unit (GRU)

Uses update and reset gates to control information retention and replacement. It provides a gated recurrent architecture with a simpler structure than an LSTM.

### Shared configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 64 |
| Hidden dimension | 128 |
| Number of recurrent layers | 1 |
| Context length | 24 tokens |
| Batch size | 32 |
| Learning rate | 0.001 |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Device | CPU |
| Training epochs | 10 |
| Training steps per epoch | 100 |

Each model uses the same main training configuration to support a more consistent comparison.

## 5. Repository Structure

```text
FGAI-Assignment-2/
├── data/
│   ├── input.txt
│   ├── processed/
│   └── processed_word/
├── models/
│   ├── rnn_best.pt
│   ├── birnn_best.pt
│   ├── lstm_best.pt
│   └── gru_best.pt
├── results/
│   ├── generated_text/
│   ├── plots/
│   ├── model_comparison.csv
│   ├── next_word_accuracy.csv
│   ├── training_summary.json
│   └── *_history.csv
├── src/
│   ├── prepare_data.py
│   ├── models.py
│   ├── train.py
│   ├── generate.py
│   ├── evaluate_accuracy.py
│   └── plot_results.py
├── .gitignore
└── README.md
```

`data/processed_word/` contains the word-level token IDs, vocabulary, and preprocessing metadata used by the current training pipeline. The `models/` directory contains the best saved checkpoints. Generated text, evaluation tables, training histories, and plots are stored under `results/`.

## 6. Installation and Setup

### Prerequisites

- Python 3.11 or a compatible Python version
- Git
- Visual Studio Code or another Python-compatible editor

### Clone the repository

```bash
git clone https://github.com/PranayDutta7/FGAI-Assignment-2.git
cd FGAI-Assignment-2
```

### Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks virtual-environment activation, use the appropriate execution policy permitted on your system or activate the environment through your IDE.

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install torch numpy matplotlib pandas
```

The project has been developed and tested using PyTorch on CPU. A CUDA-enabled GPU is not required.

## 7. Running the Project

Run these commands from the repository root with the virtual environment activated.

### Step 1: Prepare the dataset

```powershell
python .\src\prepare_data.py
```

This generates the word-level processed dataset, vocabulary, and metadata.

### Step 2: Train the models

Train all four architectures:

```powershell
python .\src\train.py --model all --epochs 10 --steps-per-epoch 100
```

To train an individual architecture:

```powershell
python .\src\train.py --model gru --epochs 10 --steps-per-epoch 100
```

Available model choices are `rnn`, `birnn`, `lstm`, `gru`, and `all`.

The training script saves model checkpoints, training histories, and a training summary.

### Step 3: Evaluate next-word prediction accuracy

```powershell
python .\src\evaluate_accuracy.py
```

This evaluates top-1 next-word prediction accuracy on eligible validation windows and updates the model comparison CSV.

### Step 4: Generate text

Generate text from the same seed at temperature 0.5:

```powershell
python .\src\generate.py --model all --seed-text "First Citizen: Before we proceed" --tokens 100 --temperature 0.5
```

Repeat at temperatures 0.8 and 1.2:

```powershell
python .\src\generate.py --model all --seed-text "First Citizen: Before we proceed" --tokens 100 --temperature 0.8
python .\src\generate.py --model all --seed-text "First Citizen: Before we proceed" --tokens 100 --temperature 1.2
```

Generated samples are saved in `results/generated_text/`.

### Step 5: Generate plots and comparison tables

```powershell
python .\src\plot_results.py
```

The plotting script produces visualisations of training loss, validation loss, validation perplexity, best validation metrics, training time, and parameter counts.

## 8. Evaluation Metrics

The models are compared using four primary quantitative metrics.

- **Validation loss:** Measures the model's average negative log-likelihood of the correct next word. Lower is better.
- **Perplexity:** Exponentiated validation loss, measuring how uncertain the model is about the next token. Lower is better.
- **Top-1 next-word accuracy:** Percentage of validation examples where the model's highest-scoring predicted word matches the actual next word. Higher is better.
- **Training time:** Recorded duration of the training run. Lower is faster under the measured configuration.

## 9. Quantitative Results

The following results were obtained from the current experiment.

| Model | Validation loss ↓ | Perplexity ↓ | Accuracy ↑ | Training time ↓ |
|---|---:|---:|---:|---:|
| Simple RNN | 6.3997 | 601.65 | **12.9381%** | **24.37 s** |
| BiRNN | 6.4291 | 619.61 | 12.0027% | 48.43 s |
| LSTM | 6.5031 | 667.24 | 12.2632% | 29.32 s |
| GRU | **6.3314** | **561.96** | 12.8355% | 39.11 s |

The top-1 accuracy evaluation used 25,336 eligible validation examples.

### Interpretation

**GRU:** Achieved the lowest validation loss and perplexity. This indicates the strongest performance under the probabilistic evaluation metrics in this experiment.

**Simple RNN:** Achieved the highest top-1 accuracy and the shortest measured training time. It provides a useful baseline with comparatively low computational cost.

**BiRNN:** Had the largest parameter count and longest training time, without outperforming the other architectures on the reported validation metrics.

**LSTM:** Did not achieve the best metrics in this particular run. Its gating mechanisms do not guarantee better performance when data, model capacity, training duration, and hyperparameters are limited.

These results reflect the current experimental configuration and should not be interpreted as a universal ranking of the architectures.

## 10. Temperature Analysis

Temperature controls how strongly the model favours high-probability words during sampling. If the predicted logits are \(z_i\), temperature \(T\) scales them before softmax:

\[
P(w_i)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}
\]

The same seed, model checkpoints, and number of generated tokens are used to compare temperatures.

### Temperature 0.5 — Lower temperature

- Favours higher-probability words.
- Produces more predictable and repetitive phrasing.
- Common dialogue patterns and expressions recur more frequently.
- Does not necessarily improve grammar or long-range coherence.

### Temperature 0.8 — Moderate temperature

- Allows a wider range of words and expressions.
- Can produce more varied dialogue and character names.
- May also introduce malformed combinations and inconsistencies.

### Temperature 1.2 — Higher temperature

- Produces a flatter sampling distribution.
- Increases the chance of selecting less probable words.
- Can increase lexical variety but also makes the output more random and less contextually consistent.

### Overall observation

Across the generated samples, lower temperature tended to produce more repetitive language, while higher temperature produced more unusual and less predictable word choices. The outputs still exhibited substantial grammatical and contextual weaknesses. No architecture was clearly superior in qualitative generation from the limited single-seed comparison.

The generated samples are available in `results/generated_text/` for direct inspection.

## 11. Limitations

- The training runs are relatively short, with 1,000 mini-batch updates per model.
- Word-level vocabularies can be large and assign `<UNK>` to words not observed during training.
- The model has a fixed context window of 24 tokens.
- Tiny Shakespeare is a specialised literary corpus, so the results may not generalise to other writing styles.
- Generated text demonstrates limited coherence, grammatical consistency, and long-range structure.
- Training time can vary with the machine, software versions, and system load.
- A more extensive study would explore additional epochs, context lengths, hidden dimensions, regularisation, and repeated runs across random seeds.

## 12. Conclusion

This project implements and compares Simple RNN, Bidirectional RNN, LSTM, and GRU architectures for word-level next-word prediction and autoregressive text generation.

In the current experiment, GRU achieved the best validation loss and perplexity, while Simple RNN achieved the highest top-1 accuracy and fastest training time. Temperature experiments demonstrated the trade-off between predictable word selection and output diversity, with higher temperatures generally introducing greater randomness.

The results highlight that quantitative next-word prediction performance and the perceived quality of generated text are related but distinct. Although the models learn statistical patterns in the dataset, their generated text shows that additional training, tuning, and model improvements would be needed for more coherent language generation.

## 13. Author and Repository

**Author:** Pranay Dutta  
**Repository:** https://github.com/PranayDutta7/FGAI-Assignment-2  
**Dataset:** Tiny Shakespeare — https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

---

*Foundations of Generative AI (ICT4403) — Assignment 2*

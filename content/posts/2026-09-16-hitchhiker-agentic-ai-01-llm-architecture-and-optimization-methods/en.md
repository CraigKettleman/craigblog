---
title: "Chapter 1 LLM Architecture and Optimization Methods"
slug: "hitchhiker-agentic-ai-01-llm-architecture-and-optimization-methods"
lang: "en"
date: "2026-09-16T00:02:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "This section covers the foundational architecture of large language models and the key optimization techniques that make training and infere…"
keywords:
  - "agentic AI"
  - "LLM"
  - "reinforcement learning"
  - "RLHF"
  - "MCP"
  - "A2A"
  - "agents"
---

<div class="hh-guide">

<aside class="hh-part">
<p class="hh-part-title">Part I Foundations</p>
</aside>

This section covers the foundational architecture of large language models and the key optimization techniques that make training and inference efficient. Topics are ordered as a curriculum: we begin with the transformer itself, then cover how to train it efficiently, how to adapt it cheaply, how to compress it, how to scale it, and how to accelerate its inference.

## How LLMs Work: An Intuitive Overview

Before diving into architectural details, let us build intuition for how a large language model transforms text into text. The entire process follows a simple pipeline: text <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> tokens <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> representations <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> tokens <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> text.

<figure id="ch1.f1"><figcaption><span class="hh-tag">Figure 1.1:</span>The LLM pipeline: text is tokenized into subword units, converted to integer IDs, embedded as dense vectors, processed through transformer layers, projected to vocabulary logits, and decoded back to text. The dashed loop shows autoregressive generation—each output token is appended to the input for the next forward pass.</figcaption></figure>

## Tokenization

Tokenization is the critical first step that converts raw text into the discrete symbols a language model operates on. The choice of tokenizer directly affects model quality, multilingual capability, and computational efficiency.

### Why Not Characters or Words?

<div class="hh-table" id="ch1.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Granularity</span></th><th class="ltx_align_left"><span>Vocab Size</span></th><th class="ltx_align_left"><span>Seq Length</span></th><th class="ltx_align_left"><span>Issues</span></th></tr></thead><tbody><tr><th class="ltx_align_left">Character</th><td class="ltx_align_left"><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>256</td><th class="ltx_align_left">Very long</th><td class="ltx_align_left">Attention cost <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow><span></span></semantics></math>; hard to learn long-range semantics</td></tr><tr><th class="ltx_align_left">Word</th><td class="ltx_align_left"><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>500K+</td><th class="ltx_align_left">Short</th><td class="ltx_align_left">Cannot handle rare/novel words; huge embedding table</td></tr><tr><th class="ltx_align_left">Subword</th><td class="ltx_align_left">32K–128K</td><th class="ltx_align_left">Moderate</th><td class="ltx_align_left">Best trade-off: short sequences, open vocabulary</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.1:</span>Trade-offs of different tokenization granularities.</p></div>

### Byte-Pair Encoding (BPE)

BPE [320] is the dominant tokenization algorithm used by GPT, Llama, Mistral, and most modern LLMs.

<figure id="ch1.f2"><img src="./fig_003_fig3.png" alt="Figure 1.2: BPE tokenization example: starting from characters, the algorithm iteratively merges the most frequent adjac" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.2:</span>BPE tokenization example: starting from characters, the algorithm iteratively merges the most frequent adjacent pairs until the word becomes a single token or the vocabulary budget is exhausted.</figcaption></figure>

### Other Tokenization Methods

<div class="hh-table" id="ch1.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Used By</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Key Idea</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">BPE</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>GPT-4 <span>[274]</span>, Llama-3 <span>[118]</span>, Mistral <span>[165]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Bottom-up merging of frequent pairs; deterministic</span></span></td></tr><tr><th class="ltx_align_left">WordPiece</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>BERT <span>[76]</span>, DistilBERT <span>[312]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Similar to BPE but maximizes likelihood of training data</span></span></td></tr><tr><th class="ltx_align_left">Unigram LM</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>SentencePiece (T5 <span>[303]</span>, XLNet <span>[405]</span>)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Top-down: start with large vocab, prune by likelihood impact</span></span></td></tr><tr><th class="ltx_align_left">Byte-level BPE</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>GPT-2 <span>[301]</span>+</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>BPE on raw bytes (no unknown tokens possible); 256 base vocab</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.2:</span>Comparison of subword tokenization algorithms.</p></div>

### Tokenization Best Practices

<ol><li>Vocabulary size matters: 32K is minimal; 128K enables better multilingual coverage and code handling. Llama-3 uses 128K tokens.</li><li>Special tokens: Always include <code>&lt;bos&gt;</code>, <code>&lt;eos&gt;</code>, <code>&lt;pad&gt;</code>, <code>&lt;unk&gt;</code>. For instruction-tuned models, add role markers (<code>&lt;|user|&gt;</code>, <code>&lt;|assistant|&gt;</code>).</li><li>Fertility: Measure tokens-per-word across languages. High fertility (many tokens per word) indicates poor coverage for that language.</li><li>Never tokenize across boundaries: Spaces, punctuation, and digits should be handled consistently. Most modern tokenizers prepend a space marker (“the”) to distinguish word-initial vs. continuation tokens.</li><li>Numbers: Consider digit-level tokenization for arithmetic tasks. “2024” as [“2”,“0”,“2”,“4”] enables digit-by-digit reasoning.</li><li>Code: Ensure whitespace (indentation) is tokenized efficiently. Llama-3 tokenizes runs of spaces as single tokens.</li></ol>

### Tokenization in Practice: HuggingFace Example

The <code>transformers</code> library provides a unified interface for all tokenizers. The following demonstrates encoding and decoding with a modern LLM tokenizer:

```
from transformers import AutoTokenizer

# Load Llama-3 tokenizer (128K vocabulary, byte-level BPE)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

text = "Reinforcement learning optimizes long-term rewards."

# Encode: text -> token IDs
token_ids = tokenizer.encode(text)
print(token_ids)
# [128000, 29934, 262, 11008, 4815, 6900, 1317, 9860, 21845, 13]

# Decode individual tokens to see subword splits
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
# ['<|begin_of_text|>', 'Re', 'inforce', 'ment', ' learning',
#  ' optimizes', ' long', '-term', ' rewards', '.']

# Decode back to text (round-trip)
reconstructed = tokenizer.decode(token_ids, skip_special_tokens=True)
assert reconstructed == text  # Perfect reconstruction

# Tokenize with attention mask (for batched inputs with padding)
batch = tokenizer(
    ["Short text.", "A much longer input sentence for comparison."],
    padding=True, return_tensors="pt"
)
print(batch.keys())  # dict_keys(['input_ids', 'attention_mask'])
```

<span class="hh-tag">Listing 1:</span>Tokenization encode/decode with HuggingFace Transformers.

### Special Tokens and Structured Prompts

Special tokens are reserved vocabulary entries that carry structural meaning rather than linguistic content. They are critical for controlling model behavior.

<div class="hh-table" id="ch1.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Token</span></th><th class="ltx_align_left"><span>Alias</span></th><th class="ltx_align_left"><span>Purpose</span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>&lt;bos&gt;</span> / <span>&lt;|begin_of_text|&gt;</span></th><td class="ltx_align_left">BOS</td><td class="ltx_align_left">Marks start of sequence</td></tr><tr><th class="ltx_align_left"><span>&lt;eos&gt;</span> / <span>&lt;|end_of_text|&gt;</span></th><td class="ltx_align_left">EOS</td><td class="ltx_align_left">Marks end of sequence; stops generation</td></tr><tr><th class="ltx_align_left"><span>&lt;|user|&gt;</span></th><td class="ltx_align_left">—</td><td class="ltx_align_left">Marks start of user turn in chat</td></tr><tr><th class="ltx_align_left"><span>&lt;|assistant|&gt;</span></th><td class="ltx_align_left">—</td><td class="ltx_align_left">Marks start of assistant turn in chat</td></tr><tr><th class="ltx_align_left"><span>&lt;pad&gt;</span></th><td class="ltx_align_left">PAD</td><td class="ltx_align_left">Fills batch to uniform length; masked in attention</td></tr><tr><th class="ltx_align_left"><span>&lt;unk&gt;</span></th><td class="ltx_align_left">UNK</td><td class="ltx_align_left">Out-of-vocabulary placeholder (rare with BPE)</td></tr><tr><th class="ltx_align_left"><span>[SEP]</span></th><td class="ltx_align_left">SEP</td><td class="ltx_align_left">Separates segments (BERT-style)</td></tr><tr><th class="ltx_align_left"><span>[CLS]</span></th><td class="ltx_align_left">CLS</td><td class="ltx_align_left">Classification token (BERT)</td></tr><tr><th class="ltx_align_left"><span>[MASK]</span></th><td class="ltx_align_left">MASK</td><td class="ltx_align_left">Masked token for MLM pretraining</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.3:</span>Common special tokens across LLM families.</p></div>

Modern chat models use special tokens to delineate conversational structure. These are not trained to carry semantic meaning—they are structural delimiters that the model learns to parse:

```
# Llama-3 chat template
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain PPO in one sentence."},
]

# apply_chat_template handles all special token insertion
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
print(prompt)
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# Explain PPO in one sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
#
```

<span class="hh-tag">Listing 2:</span>Chat template with special tokens (Llama-3 format).

## The Transformer Architecture

The Transformer [357] is the foundation of all modern LLMs. Understanding its components is essential for grasping every optimization and training method in this guide.

### High-Level Structure

A decoder-only transformer processes tokens sequentially through embedding, repeated attention+FFN blocks, and a final projection to vocabulary logits. Figure 图 1.3 shows the complete architecture.

<figure id="ch1.f3"><img src="./fig_004_decoder-only.png" alt="Figure 1.3: Decoder-only Transformer block (GPT-style, Pre-Norm variant). Each sub-layer (attention, FFN) is preceded by" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.3:</span>Decoder-only Transformer block (GPT-style, Pre-Norm variant). Each sub-layer (attention, FFN) is preceded by LayerNorm and followed by a residual addition: <math alttext="\mathbf{x}+\text{SubLayer}(\text{LN}(\mathbf{x}))" display="inline"><semantics><mrow><mi>𝐱</mi><mo>+</mo><mrow><mtext>SubLayer</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mtext>LN</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>. This Pre-Norm ordering (used by Llama, GPT-3, Mistral) stabilizes training without warmup, unlike the original Post-Norm (which applies LayerNorm after the addition). <math alttext="L" display="inline"><semantics><mi>L</mi></semantics></math> identical blocks are stacked, followed by a final LayerNorm and linear projection to vocabulary logits.</figcaption></figure>

### The Original Encoder-Decoder Transformer

The Transformer was originally introduced [357] as an encoder-decoder architecture for sequence-to-sequence tasks (machine translation, summarization). While modern LLMs predominantly use decoder-only variants (GPT-style), understanding the full architecture is essential because cross-attention and masked self-attention — both originating here — remain fundamental building blocks.

<figure id="ch1.f4"><img src="./fig_005_transformer-original.png" alt="Figure 1.4: The original Transformer architecture (Vaswani et al., 2017). The encoder (left) processes the full input wi" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.4:</span>The original Transformer architecture (Vaswani et al., 2017). The encoder (left) processes the full input with bidirectional self-attention. The decoder (right) generates tokens autoregressively using masked self-attention and cross-attention to encoder representations. Dashed boxes indicate the repeated layer block (<math alttext="\times N" display="inline"><semantics><mrow><mphantom></mphantom><mo lspace="0.222em" rspace="0.222em">×</mo><mi>N</mi></mrow></semantics></math>); gray lines show residual connections bypassing each sub-layer. Note: the original work uses Post-Norm (LayerNorm applied <em>after</em> the residual addition: <math alttext="\text{LN}(\mathbf{x}+\text{SubLayer}(\mathbf{x}))" display="inline"><semantics><mrow><mtext>LN</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>𝐱</mi><mo>+</mo><mrow><mtext>SubLayer</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>), unlike modern LLMs which use Pre-Norm.</figcaption></figure>

The encoder processes the entire input sequence <em>bidirectionally</em> — each token attends to all other tokens (no causal mask). This produces a rich contextual representation <math alttext="\mathbf{H}^{\text{enc}}\in\mathbb{R}^{n\times d}" display="inline"><semantics><mrow><msup><mi>𝐇</mi><mtext>enc</mtext></msup><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math> where each position encodes information about the full input:

<ul><li>Input: Token embeddings + sinusoidal positional encodings</li><li>Each layer: Multi-Head Self-Attention <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Add &amp; Norm <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> FFN <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Add &amp; Norm</li><li>No causal mask: Position <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> attends to all positions <math alttext="1,\ldots,n" display="inline"><semantics><mrow><mn>1</mn><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mi>n</mi></mrow></semantics></math></li><li>Output: Contextual representations of the full input sequence</li></ul>

The decoder generates output tokens one at a time (autoregressively). To prevent the model from “seeing the future,” the self-attention in the decoder uses a causal mask:

<div class="hh-equation" id="ch1.e1"><math alttext="\text{MaskedAttn}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}+M\right)V" display="block"><semantics><mrow><mrow><mtext>MaskedAttn</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mrow><mfrac><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>+</mo><mi>M</mi></mrow><mo>)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(1.1)</span></div>

where the mask <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math> is:

<div class="hh-equation" id="ch1.ex1"><math alttext="M_{ij}=\begin{cases}0&amp;\text{if }i\geq j\text{ (can attend)}\\
-\infty&amp;\text{if }i&lt;j\text{ (future token --- blocked)}\end{cases}" display="block"><semantics><mrow><msub><mi>M</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mn>0</mn></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>≥</mo><mrow><mi>j</mi><mo lspace="0em" rspace="0em">​</mo><mtext> (can attend)</mtext></mrow></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>&lt;</mo><mrow><mi>j</mi><mo lspace="0em" rspace="0em">​</mo><mtext> (future token — blocked)</mtext></mrow></mrow></mtd></mtr></mtable></mrow></mrow></semantics></math></div>

After masked self-attention, each decoder layer applies cross-attention where the decoder attends to the encoder’s output representations. This is the mechanism by which the decoder “reads” the input:

<div class="hh-equation" id="ch1.e2"><math alttext="\text{CrossAttn}(Q_{\text{dec}},K_{\text{enc}},V_{\text{enc}})=\text{softmax}\!\left(\frac{Q_{\text{dec}}K_{\text{enc}}^{T}}{\sqrt{d_{k}}}\right)V_{\text{enc}}" display="block"><semantics><mrow><mrow><mtext>CrossAttn</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>Q</mi><mtext>dec</mtext></msub><mo>,</mo><msub><mi>K</mi><mtext>enc</mtext></msub><mo>,</mo><msub><mi>V</mi><mtext>enc</mtext></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>Q</mi><mtext>dec</mtext></msub><mo lspace="0em" rspace="0em">​</mo><msubsup><mi>K</mi><mtext>enc</mtext><mi>T</mi></msubsup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>V</mi><mtext>enc</mtext></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(1.2)</span></div>

<ul><li>Queries come from the decoder’s previous sublayer (the masked self-attention output)</li><li>Keys and Values come from the encoder’s final output <math alttext="\mathbf{H}^{\text{enc}}" display="inline"><semantics><msup><mi>𝐇</mi><mtext>enc</mtext></msup></semantics></math></li><li>No mask is applied — every decoder position can attend to every encoder position</li><li>This allows the decoder to dynamically focus on different parts of the input at each generation step (e.g., attending to “cat” when translating to “gato” in English<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>Spanish)</li></ul>

Each decoder layer contains three sublayers (vs. two in the encoder):

<ol><li>Masked Multi-Head Self-Attention + Residual + LayerNorm</li><li>Multi-Head Cross-Attention (to encoder output) + Residual + LayerNorm</li><li>Feed-Forward Network + Residual + LayerNorm</li></ol>

Modern LLMs (GPT, Llama, Qwen) use only the decoder, removing both the encoder and cross-attention layers entirely. The key insight: for generative language modeling, a single causal (masked) self-attention stack is sufficient — the model learns to encode context and generate continuations in a single pass. This simplifies architecture, training, and inference while scaling more effectively. Encoder-decoder models (T5, BART) remain relevant for tasks with distinct input/output structure (translation, summarization), and cross-attention reappears in multimodal models where vision encoders provide keys/values to language decoders.

### Decoder-Only vs Encoder-Decoder

Modern LLMs almost exclusively use decoder-only architectures, but understanding the trade-offs with encoder-decoder designs clarifies why.

ArchitectureExamplesUse CaseDecoder-onlyGPT-4 [274], Llama [118], Mistral [165], Qwen [350]Autoregressive generation; dominant for chat/reasoningEncoder-decoderT5 [303], BART [207], Flan-T5 [59]Seq2seq (translation, summarization); less common nowEncoder-onlyBERT [76], RoBERTa [234]Classification/embeddings; not for generation

### Embeddings: From Discrete Tokens to Continuous Space

Before any attention or computation happens, the transformer must convert discrete token IDs into continuous vectors that neural networks can process. This is the role of the embedding layer.

An embedding is a learned dense vector representation of a discrete symbol. Instead of representing the word “king” as a one-hot vector of size <math alttext="|\mathcal{V}|=128{,}000" display="inline"><semantics><mrow><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow><mo>=</mo><mn>128,000</mn></mrow></semantics></math> (mostly zeros), we represent it as a compact vector in <math alttext="\mathbb{R}^{d}" display="inline"><semantics><msup><mi>ℝ</mi><mi>d</mi></msup></semantics></math> (e.g., <math alttext="d=4096" display="inline"><semantics><mrow><mi>d</mi><mo>=</mo><mn>4096</mn></mrow></semantics></math>) that captures its <em>meaning</em>.

The key insight: similar concepts get nearby vectors. In a well-trained embedding space:

<ul><li>“king” and “queen” are close (both royalty)</li><li>“king” and “bicycle” are far apart (unrelated)</li><li>Vector arithmetic captures relationships: <math alttext="\vec{\text{king}}-\vec{\text{man}}+\vec{\text{woman}}\approx\vec{\text{queen}}" display="inline"><semantics><mrow><mrow><mrow><mover accent="true"><mtext>king</mtext><mo stretchy="false">→</mo></mover><mo>−</mo><mover accent="true"><mtext>man</mtext><mo stretchy="false">→</mo></mover></mrow><mo>+</mo><mover accent="true"><mtext>woman</mtext><mo stretchy="false">→</mo></mover></mrow><mo>≈</mo><mover accent="true"><mtext>queen</mtext><mo stretchy="false">→</mo></mover></mrow></semantics></math></li></ul>

<figure id="ch1.f5"><img src="./fig_006_fig6.png" alt="Figure 1.5: Embedding space visualization (2D projection): semantically similar words cluster together. The embedding ta" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.5:</span>Embedding space visualization (2D projection): semantically similar words cluster together. The embedding table learns these positions during pretraining, capturing meaning purely from co-occurrence patterns in text.</figcaption></figure>

In practice, the embedding layer is simply a matrix <math alttext="\mathbf{E}\in\mathbb{R}^{|\mathcal{V}|\times d}" display="inline"><semantics><mrow><mi>𝐄</mi><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo rspace="0.055em" stretchy="false">|</mo></mrow><mo rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math> where row <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> stores the embedding vector for token <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math>:

<div class="hh-equation" id="ch1.e3"><math alttext="\text{embed}(x_{t})=\mathbf{E}[x_{t}]\in\mathbb{R}^{d}" display="block"><semantics><mrow><mrow><mtext>embed</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>𝐄</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">]</mo></mrow></mrow><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math><span class="hh-equation-number">(1.3)</span></div>

For a sequence of token IDs <math alttext="[x_{1},x_{2},\ldots,x_{n}]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><msub><mi>x</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>x</mi><mi>n</mi></msub></mrow><mo stretchy="false">]</mo></mrow></semantics></math>, embedding is a simple table lookup (indexing operation):

<div class="hh-equation" id="ch1.ex2"><math alttext="\mathbf{H}_{0}=[\mathbf{E}[x_{1}];\;\mathbf{E}[x_{2}];\;\ldots;\;\mathbf{E}[x_{n}]]\in\mathbb{R}^{n\times d}" display="block"><semantics><mrow><msub><mi>𝐇</mi><mn>0</mn></msub><mo>=</mo><mrow><mo stretchy="false">[</mo><mrow><mrow><mi>𝐄</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">]</mo></mrow></mrow><mo rspace="0.447em">;</mo><mrow><mi>𝐄</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">]</mo></mrow></mrow><mo rspace="0.447em">;</mo><mi mathvariant="normal">…</mi><mo rspace="0.447em">;</mo><mrow><mi>𝐄</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><msub><mi>x</mi><mi>n</mi></msub><mo stretchy="false">]</mo></mrow></mrow></mrow><mo stretchy="false">]</mo></mrow><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math></div>

A critical issue arises when using pretrained embeddings (e.g., from BERT or GPT-2) for downstream tasks like retrieval (RAG) or bootstrapping recommender systems: the learned representations are highly anisotropic—they occupy a narrow cone in the embedding space rather than being uniformly distributed across all directions [87].

<figure id="ch1.f6"><img src="./fig_007_fig7.png" alt="Figure 1.6: Isotropy vs. anisotropy in embedding spaces. Left: isotropic embeddings spread uniformly, making cosine simi" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.6:</span>Isotropy vs. anisotropy in embedding spaces. Left: isotropic embeddings spread uniformly, making cosine similarity a reliable measure of semantic relatedness. Right: anisotropic embeddings (as found in BERT) cluster in a narrow cone, causing all pairs to have high cosine similarity regardless of semantic content. Whitening transforms the space to restore isotropy.</figcaption></figure>

Why this matters for applications:

<ul><li>RAG / Retrieval: If all embeddings have cosine similarity <math alttext="&gt;0.7" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mn>0.7</mn></mrow></semantics></math> regardless of content, retrieval rankings become nearly random—the system cannot distinguish relevant from irrelevant passages.</li><li>Recommender systems: Using pretrained LLM embeddings to represent items/users only works if the geometry preserves meaningful similarity structure.</li><li>Clustering: Anisotropic embeddings collapse clusters, making it impossible to discover natural groupings.</li></ul>

A simple and effective fix is whitening[341]—a linear transformation that makes the embedding distribution isotropic (zero mean, identity covariance):

<div class="hh-equation" id="ch1.e4"><math alttext="\tilde{\mathbf{h}}=\mathbf{D}^{-1/2}\mathbf{U}^{T}(\mathbf{h}-\bm{\mu})" display="block"><semantics><mrow><mover accent="true"><mi>𝐡</mi><mo>~</mo></mover><mo>=</mo><msup><mi>𝐃</mi><mrow><mo>−</mo><mn>1</mn><mo>/</mo><mn>2</mn></mrow></msup><msup><mi>𝐔</mi><mi>T</mi></msup><mrow><mo stretchy="false">(</mo><mi>𝐡</mi><mo>−</mo><mi>𝝁</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(1.4)</span></div>

where <math alttext="\bm{\mu}" display="inline"><semantics><mi>𝝁</mi></semantics></math> is the mean embedding, and <math alttext="\mathbf{U}\mathbf{D}\mathbf{U}^{T}" display="inline"><semantics><msup><mi>𝐔𝐃𝐔</mi><mi>T</mi></msup></semantics></math> is the eigendecomposition of the covariance matrix <math alttext="\Sigma=\frac{1}{N}\sum_{i}(\mathbf{h}_{i}-\bm{\mu})(\mathbf{h}_{i}-\bm{\mu})^{T}" display="inline"><semantics><mrow><mi mathvariant="normal">Σ</mi><mo>=</mo><mrow><mfrac><mn>1</mn><mi>N</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo rspace="0em">∑</mo><mi>i</mi></msub><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>𝐡</mi><mi>i</mi></msub><mo>−</mo><mi>𝝁</mi></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mrow><msub><mi>𝐡</mi><mi>i</mi></msub><mo>−</mo><mi>𝝁</mi></mrow><mo stretchy="false">)</mo></mrow><mi>T</mi></msup></mrow></mrow></mrow></mrow></semantics></math>.

### Self-Attention Mechanism

Self-attention is the core operation that allows each token to attend to every other token in the sequence, computing a weighted combination based on relevance.

The naive attention computation has quadratic cost in sequence length:

<ul><li>Time: <math alttext="O(n^{2}\cdot d)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msup><mi>n</mi><mn>2</mn></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>d</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> — computing <math alttext="QK^{T}" display="inline"><semantics><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mi>T</mi></msup></mrow></semantics></math> requires <math alttext="n^{2}" display="inline"><semantics><msup><mi>n</mi><mn>2</mn></msup></semantics></math> dot products, each of dimension <math alttext="d_{k}" display="inline"><semantics><msub><mi>d</mi><mi>k</mi></msub></semantics></math>.</li><li>Memory: <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> — the full attention matrix must be materialized to apply softmax.</li></ul>

For a 128K-token context with <math alttext="d=4096" display="inline"><semantics><mrow><mi>d</mi><mo>=</mo><mn>4096</mn></mrow></semantics></math>, the attention matrix alone is <math alttext="128\text{K}\times 128\text{K}=16.4" display="inline"><semantics><mrow><mrow><mrow><mrow><mn>128</mn><mo lspace="0em" rspace="0em">​</mo><mtext>K</mtext></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mn>128</mn></mrow><mo lspace="0em" rspace="0em">​</mo><mtext>K</mtext></mrow><mo>=</mo><mn>16.4</mn></mrow></semantics></math> billion entries (64 GB in FP32). This quadratic scaling is the fundamental bottleneck for long-context LLMs.

<div class="hh-table" id="ch1.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Seq Length</span></th><th class="ltx_align_left"><span>Attention Ops</span></th><th class="ltx_align_left"><span>Matrix Size</span></th><th class="ltx_align_left"><span>Practical Impact</span></th></tr></thead><tbody><tr><td class="ltx_align_left">2K</td><td class="ltx_align_left">4M</td><td class="ltx_align_left">16 MB</td><td class="ltx_align_left">Fast; fits in SRAM</td></tr><tr><td class="ltx_align_left">8K</td><td class="ltx_align_left">64M</td><td class="ltx_align_left">256 MB</td><td class="ltx_align_left">Manageable with FlashAttention</td></tr><tr><td class="ltx_align_left">32K</td><td class="ltx_align_left">1B</td><td class="ltx_align_left">4 GB</td><td class="ltx_align_left">Requires memory-efficient kernels</td></tr><tr><td class="ltx_align_left">128K</td><td class="ltx_align_left">16B</td><td class="ltx_align_left">64 GB</td><td class="ltx_align_left">Exceeds single GPU HBM</td></tr><tr><td class="ltx_align_left">1M</td><td class="ltx_align_left">1T</td><td class="ltx_align_left">4 TB</td><td class="ltx_align_left">Impossible without sub-quadratic methods</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.4:</span>Attention cost scaling: why naive implementation is prohibitive for long sequences.</p></div>

Several families of solutions address this quadratic bottleneck:

<ol><li>Exact attention with IO-awareness (FlashAttention [64]): Does not reduce computational complexity but eliminates the need to materialize the <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> matrix in HBM by computing attention in tiles that fit in SRAM. Crucially, FlashAttention is orthogonal to the sparse patterns below—it is an execution engine, not an attention pattern. Production systems routinely combine FlashAttention with sliding windows or block-sparse masks, getting both IO efficiency and reduced FLOPs. We cover the algorithm in detail in Section 1.6 Flash Attention – Algorithm and Hardware Awareness.</li><li>Sliding window / local attention: Each token only attends to the <math alttext="w" display="inline"><semantics><mi>w</mi></semantics></math> nearest tokens (e.g., <math alttext="w=4096" display="inline"><semantics><mrow><mi>w</mi><mo>=</mo><mn>4096</mn></mrow></semantics></math>). Cost becomes <math alttext="O(n\cdot w)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>w</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>—linear in <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>. Used by Mistral [165] (window <math alttext="=4096" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mn>4096</mn></mrow></semantics></math>) and Longformer [23]. Trades global context for efficiency; works well because most attention is local in practice. In modern stacks, the sliding-window mask is executed <em>inside</em> a FlashAttention kernel.</li><li>Sparse attention patterns: Combine local windows with periodic global tokens (e.g., every 512th token attends to all). BigBird [416] and LongT5 [125] use this. Preserves some long-range connectivity at <math alttext="O(n\sqrt{n})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo lspace="0em" rspace="0em">​</mo><msqrt><mi>n</mi></msqrt></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> cost. Again, FlashAttention serves as the underlying kernel for the non-zero attention blocks.</li><li>Linear attention / state-space models: Replace <math alttext="\text{softmax}(QK^{T})V" display="inline"><semantics><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mi>T</mi></msup></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></semantics></math> with <math alttext="\phi(Q)(\phi(K)^{T}V)" display="inline"><semantics><mrow><mi>ϕ</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>Q</mi><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>ϕ</mi><mo lspace="0em" rspace="0em">​</mo><msup><mrow><mo stretchy="false">(</mo><mi>K</mi><mo stretchy="false">)</mo></mrow><mi>T</mi></msup><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> using associativity, or reformulate as a recurrence (Mamba [122], RWKV [288]). Theoretically <math alttext="O(n\cdot d^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msup><mi>d</mi><mn>2</mn></msup></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> total. Unlike approaches 2–3 above, these are <em>architectural replacements</em> that alter model expressiveness—softmax-free attention is fundamentally less expressive, and empirically these models still lag behind transformers on tasks requiring precise long-range retrieval or complex reasoning.</li><li>KV cache compression: At inference, compress or evict old KV pairs to bound memory. Techniques include: H<sub>2</sub>O [429] (heavy-hitter oracle—keep only high-attention keys), StreamingLLM [394] (keep initial “attention sink” tokens + recent window), and quantized KV caches [238].</li></ol>

### Multi-Head Attention

Rather than computing a single attention function, multi-head attention runs several attention operations in parallel, each learning to focus on different aspects of the input (syntax, semantics, position, etc.).

DeepSeek-V2 [70] introduced a more aggressive compression: instead of sharing KV heads (as in GQA), MLA compresses <em>all</em> key-value information into a single low-rank latent vector per token:

<div class="hh-equation" id="ch1.ex6"><math alttext="c_{t}=W_{DKV}\,h_{t},\quad K_{t}=W_{UK}\,c_{t},\quad V_{t}=W_{UV}\,c_{t}" display="block"><semantics><mrow><mrow><msub><mi>c</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>W</mi><mrow><mi>D</mi><mo lspace="0em" rspace="0em">​</mo><mi>K</mi><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>h</mi><mi>t</mi></msub></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>K</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>W</mi><mrow><mi>U</mi><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>c</mi><mi>t</mi></msub></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>V</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>W</mi><mrow><mi>U</mi><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></msub><mo lspace="0em" rspace="0em">​</mo><msub><mi>c</mi><mi>t</mi></msub></mrow></mrow></mrow></semantics></math></div>

where <math alttext="c_{t}\in\mathbb{R}^{d_{c}}" display="inline"><semantics><mrow><msub><mi>c</mi><mi>t</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><msub><mi>d</mi><mi>c</mi></msub></msup></mrow></semantics></math> with <math alttext="d_{c}\ll n_{h}\cdot d_{h}" display="inline"><semantics><mrow><msub><mi>d</mi><mi>c</mi></msub><mo>≪</mo><mrow><msub><mi>n</mi><mi>h</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>d</mi><mi>h</mi></msub></mrow></mrow></semantics></math>. The KV cache stores only <math alttext="c_{t}" display="inline"><semantics><msub><mi>c</mi><mi>t</mi></msub></semantics></math>—smaller than even GQA—while decompression matrices <math alttext="W_{UK},W_{UV}" display="inline"><semantics><mrow><msub><mi>W</mi><mrow><mi>U</mi><mo lspace="0em" rspace="0em">​</mo><mi>K</mi></mrow></msub><mo>,</mo><msub><mi>W</mi><mrow><mi>U</mi><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></msub></mrow></semantics></math> reconstruct full keys and values on-the-fly. Because the compression is learned end-to-end, quality matches or exceeds GQA. MLA enables DeepSeek’s subsequent Sparse Attention for sub-quadratic long-context scaling and is now the basis of the DeepSeek-V3/V4 architecture family.

### Positional Encodings

Transformers are permutation-equivariant by construction — without positional information, the model cannot distinguish “the cat sat on the mat” from “mat the on sat cat the”. Positional encodings inject sequence-order signal so that attention can reason about token distance and direction.

<div class="hh-table" id="ch1.t5"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Used By</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Key Idea</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Sinusoidal</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Original Transformer</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Fixed <math alttext="\sin/\cos" display="inline"><semantics><mrow><mi>sin</mi><mo lspace="0em">/</mo><mi>cos</mi></mrow><span></span></semantics></math> at different frequencies. Not learned.</span></span></td></tr><tr><th class="ltx_align_left">Learned Absolute</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>GPT-2 <span>[301]</span>, BERT <span>[76]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learned embedding per position. Limited to training length.</span></span></td></tr><tr><th class="ltx_align_left">RoPE (Rotary)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Llama <span>[118]</span>, Qwen <span>[350]</span>, Mistral <span>[165]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Rotate Q,K vectors by position-dependent angle. Extrapolates via NTK-aware scaling.</span></span></td></tr><tr><th class="ltx_align_left">ALiBi</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>BLOOM <span>[383]</span>, MPT <span>[260]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No position embedding; add linear bias <math alttext="-m|i-j|" display="inline"><semantics><mrow><mo>−</mo><mrow><mi>m</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><mrow><mi>i</mi><mo>−</mo><mi>j</mi></mrow><mo stretchy="false">|</mo></mrow></mrow></mrow><span></span></semantics></math> to attention scores. Simple, extrapolates well.</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.5:</span>Positional encoding methods in modern LLMs.</p></div>

Introduced in the original Transformer [357], this method uses fixed sinusoidal functions at geometrically-spaced frequencies:

<div class="hh-equation" id="ch1.ex7"><math alttext="\text{PE}(pos,2i)=\sin\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr),\qquad\text{PE}(pos,2i{+}1)=\cos\!\Bigl(\frac{pos}{10000^{2i/d}}\Bigr)" display="block"><semantics><mrow><mrow><mrow><mtext>PE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo>,</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.243em"><mi>sin</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.600em" minsize="1.600em">(</mo><mfrac><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><msup><mn>10000</mn><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>/</mo><mi>d</mi></mrow></msup></mfrac><mo maxsize="1.600em" minsize="1.600em">)</mo></mrow></mrow></mrow><mo rspace="2.167em">,</mo><mrow><mrow><mtext>PE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo>,</mo><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>+</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="1.216em"><mi>cos</mi></mpadded><mo>⁡</mo><mrow><mo maxsize="1.600em" minsize="1.600em">(</mo><mfrac><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><msup><mn>10000</mn><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>/</mo><mi>d</mi></mrow></msup></mfrac><mo maxsize="1.600em" minsize="1.600em">)</mo></mrow></mrow></mrow></mrow></semantics></math></div>

where <math alttext="pos" display="inline"><semantics><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow></semantics></math> is the token position, <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> is the dimension index, and <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> is the model dimension.

Motivation: Each frequency encodes position at a different scale (analogous to binary counting). The authors hypothesised that the model could learn to attend to relative positions because <math alttext="\text{PE}(pos+k)" display="inline"><semantics><mrow><mtext>PE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo>+</mo><mi>k</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> can be expressed as a linear function of <math alttext="\text{PE}(pos)" display="inline"><semantics><mrow><mtext>PE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>.

Pros: Zero learned parameters; deterministic; theoretically supports arbitrary lengths. <br>

Cons: In practice, does not extrapolate well beyond training lengths; the model must learn to decode relative position from absolute signals indirectly; largely superseded.

Used by GPT-2 [301] and BERT [76]: a learnable embedding matrix <math alttext="\mathbf{E}_{\text{pos}}\in\mathbb{R}^{L_{\max}\times d}" display="inline"><semantics><mrow><msub><mi>𝐄</mi><mtext>pos</mtext></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><msub><mi>L</mi><mi>max</mi></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math> is added to token embeddings:

<div class="hh-equation" id="ch1.ex8"><math alttext="h_{0}^{(pos)}=\text{TokenEmbed}(x_{pos})+\mathbf{E}_{\text{pos}}[pos]" display="block"><semantics><mrow><msubsup><mi>h</mi><mn>0</mn><mrow><mo stretchy="false">(</mo><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo stretchy="false">)</mo></mrow></msubsup><mo>=</mo><mrow><mrow><mtext>TokenEmbed</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mrow><msub><mi>𝐄</mi><mtext>pos</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><mi>p</mi><mo lspace="0em" rspace="0em">​</mo><mi>o</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></semantics></math></div>

Motivation: Let the model learn whatever positional representation is optimal for the task, rather than imposing a fixed structure.

Pros: Maximum flexibility; simple implementation; often outperforms sinusoidal for short sequences. <br>

Cons: Hard-coded maximum length <math alttext="L_{\max}" display="inline"><semantics><msub><mi>L</mi><mi>max</mi></msub></semantics></math>; no generalisation beyond it; embeddings near the end of <math alttext="L_{\max}" display="inline"><semantics><msub><mi>L</mi><mi>max</mi></msub></semantics></math> are under-trained; adds <math alttext="L_{\max}\times d" display="inline"><semantics><mrow><msub><mi>L</mi><mi>max</mi></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></semantics></math> parameters.

RoPE [340] encodes position by <em>rotating</em> query and key vectors in 2D subspaces:

<div class="hh-equation" id="ch1.ex9"><math alttext="\text{RoPE}(x_{m},m)=\begin{pmatrix}x_{m}^{(1)}\\
x_{m}^{(2)}\\
\vdots\\
x_{m}^{(d-1)}\\
x_{m}^{(d)}\end{pmatrix}\odot\begin{pmatrix}\cos m\theta_{1}\\
\cos m\theta_{1}\\
\vdots\\
\cos m\theta_{d/2}\\
\cos m\theta_{d/2}\end{pmatrix}+\begin{pmatrix}-x_{m}^{(2)}\\
x_{m}^{(1)}\\
\vdots\\
-x_{m}^{(d)}\\
x_{m}^{(d-1)}\end{pmatrix}\odot\begin{pmatrix}\sin m\theta_{1}\\
\sin m\theta_{1}\\
\vdots\\
\sin m\theta_{d/2}\\
\sin m\theta_{d/2}\end{pmatrix}" display="block"><semantics><mrow><mrow><mtext>RoPE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>m</mi></msub><mo>,</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mrow><mo>(</mo><mtable displaystyle="true" rowspacing="0pt"><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr><mtr><mtd></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mrow><mi>d</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr></mtable><mo rspace="0.055em">)</mo></mrow><mo rspace="0.222em">⊙</mo><mrow><mo>(</mo><mtable displaystyle="true" rowspacing="0pt"><mtr><mtd><mrow><mrow><mi>cos</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mtd></mtr><mtr><mtd><mrow><mrow><mi>cos</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mtd></mtr><mtr><mtd></mtd></mtr><mtr><mtd><mrow><mrow><mi>cos</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mrow><mi>d</mi><mo>/</mo><mn>2</mn></mrow></msub></mrow></mtd></mtr><mtr><mtd><mrow><mrow><mi>cos</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mrow><mi>d</mi><mo>/</mo><mn>2</mn></mrow></msub></mrow></mtd></mtr></mtable><mo>)</mo></mrow></mrow><mo>+</mo><mrow><mrow><mo>(</mo><mtable displaystyle="true" rowspacing="0pt"><mtr><mtd><mrow><mo>−</mo><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mn>2</mn><mo stretchy="false">)</mo></mrow></msubsup></mrow></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr><mtr><mtd></mtd></mtr><mtr><mtd><mrow><mo>−</mo><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mi>d</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow></mtd></mtr><mtr><mtd><msubsup><mi>x</mi><mi>m</mi><mrow><mo stretchy="false">(</mo><mrow><mi>d</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></msubsup></mtd></mtr></mtable><mo rspace="0.055em">)</mo></mrow><mo rspace="0.222em">⊙</mo><mrow><mo>(</mo><mtable displaystyle="true" rowspacing="0pt"><mtr><mtd><mrow><mrow><mi>sin</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mtd></mtr><mtr><mtd><mrow><mrow><mi>sin</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mn>1</mn></msub></mrow></mtd></mtr><mtr><mtd></mtd></mtr><mtr><mtd><mrow><mrow><mi>sin</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mrow><mi>d</mi><mo>/</mo><mn>2</mn></mrow></msub></mrow></mtd></mtr><mtr><mtd><mrow><mrow><mi>sin</mi><mo lspace="0.167em">⁡</mo><mi>m</mi></mrow><mo lspace="0em" rspace="0em">​</mo><msub><mi>θ</mi><mrow><mi>d</mi><mo>/</mo><mn>2</mn></mrow></msub></mrow></mtd></mtr></mtable><mo>)</mo></mrow></mrow></mrow></mrow></semantics></math></div>

where <math alttext="\theta_{i}=10000^{-2i/d}" display="inline"><semantics><mrow><mi>θ</mi><msub><mrow></mrow><mi>i</mi></msub><mo>=</mo><mn>10000</mn><msup><mrow></mrow><mrow><mo>−</mo><mn>2</mn><mi>i</mi><mo>/</mo><mi>d</mi></mrow></msup></mrow></semantics></math> and <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> is the position index. The key property is that the dot product between rotated queries and keys depends only on relative position:

<div class="hh-equation" id="ch1.ex10"><math alttext="\langle\text{RoPE}(q_{m},m),\;\text{RoPE}(k_{n},n)\rangle=f(q_{m},k_{n},m-n)" display="block"><semantics><mrow><mrow><mo stretchy="false">⟨</mo><mrow><mrow><mtext>RoPE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>m</mi></msub><mo>,</mo><mi>m</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.447em">,</mo><mrow><mtext>RoPE</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>k</mi><mi>n</mi></msub><mo>,</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">⟩</mo></mrow><mo>=</mo><mrow><mi>f</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>q</mi><mi>m</mi></msub><mo>,</mo><msub><mi>k</mi><mi>n</mi></msub><mo>,</mo><mrow><mi>m</mi><mo>−</mo><mi>n</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math></div>

Motivation: Achieve relative position encoding without explicit bias terms, while maintaining compatibility with linear attention and KV-caching.

Pros: Naturally relative; no extra parameters; compatible with efficient inference; can be extended to longer contexts via NTK-aware scaling [289] or YaRN (adjusting <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> base or interpolating frequencies). <br>

Cons: Slightly more compute per attention operation (rotation + interleaving); extrapolation requires explicit scaling strategies; rotation in 2D subspaces imposes structure that may not be optimal for all tasks.

ALiBi [292] takes a radically different approach: <em>no positional embedding at all</em>. Instead, a static linear penalty is subtracted from attention scores:

<div class="hh-equation" id="ch1.ex11"><math alttext="\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}-m\cdot\bigl[|i-j|\bigr]_{i,j}\right)V" display="block"><semantics><mrow><mrow><mtext>Attention</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mrow><mfrac><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>−</mo><mrow><mi>m</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mrow><mo maxsize="1.200em" minsize="1.200em">[</mo><mrow><mo stretchy="false">|</mo><mrow><mi>i</mi><mo>−</mo><mi>j</mi></mrow><mo stretchy="false">|</mo></mrow><mo maxsize="1.200em" minsize="1.200em">]</mo></mrow><mrow><mi>i</mi><mo>,</mo><mi>j</mi></mrow></msub></mrow></mrow><mo>)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></mrow></semantics></math></div>

where <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> is a head-specific slope (set geometrically: <math alttext="m_{h}=2^{-8h/H}" display="inline"><semantics><mrow><mi>m</mi><msub><mrow></mrow><mi>h</mi></msub><mo>=</mo><mn>2</mn><msup><mrow></mrow><mrow><mo>−</mo><mn>8</mn><mi>h</mi><mo>/</mo><mi>H</mi></mrow></msup></mrow></semantics></math> for head <math alttext="h" display="inline"><semantics><mi>h</mi></semantics></math> of <math alttext="H" display="inline"><semantics><mi>H</mi></semantics></math> total). The bias <math alttext="-m|i-j|" display="inline"><semantics><mrow><mo>−</mo><mrow><mi>m</mi><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><mrow><mi>i</mi><mo>−</mo><mi>j</mi></mrow><mo stretchy="false">|</mo></mrow></mrow></mrow></semantics></math> creates a soft local attention window whose width varies by head.

Motivation: Position should bias attention toward nearby tokens (recency prior) without interfering with the embedding space. By operating purely in attention-score space, ALiBi avoids polluting token representations with positional signal.

Pros: Excellent length extrapolation (trained at 1k, works at 8k+); zero parameters; trivial to implement; head-specific slopes give multi-scale locality. <br>

Cons: Less expressive for tasks requiring precise long-range positional reasoning (e.g. “what was the 5th word?”); the linear decay is a strong inductive bias that may not suit all domains; largely overtaken by RoPE in recent models due to RoPE’s better short-context performance.

<div class="hh-table" id="ch1.t6"><table class="ltx_tabular ltx_align_middle"><thead><tr><th></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Sinusoidal</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Learned Abs.</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>RoPE</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>ALiBi</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Extra parameters</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="L_{\max}\times d" display="inline"><semantics><mrow><msub><mi>L</mi><mi>max</mi></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None</span></span></td></tr><tr><th class="ltx_align_left">Position type</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Absolute</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Absolute</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Relative</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Relative (implicit)</span></span></td></tr><tr><th class="ltx_align_left">Length extrapolation</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Poor</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Good (w/ scaling)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Excellent</span></span></td></tr><tr><th class="ltx_align_left">Compute overhead</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Negligible</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Negligible</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Small</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Negligible</span></span></td></tr><tr><th class="ltx_align_left">Dominant era</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2017–19</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2018–20</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2022–present</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2022–23</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.6:</span>Positional encoding comparison: practical trade-offs.</p></div>

Modern frontier models (Claude [11] with 200K–1M context, Gemini 1.5 [108] at 1M+, GPT-4 [274] at 128K) require positional encodings that remain faithful far beyond training lengths. The dominant solutions today:

<ol><li>RoPE with frequency scaling: The standard approach for extending RoPE beyond training length. Rather than retraining, the base frequency <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> is rescaled:<math alttext="\theta^{\prime}_{i}=\theta_{i}\cdot\left(\frac{L_{\text{target}}}{L_{\text{train}}}\right)^{2i/d}" display="block"><semantics><mrow><msubsup><mi>θ</mi><mi>i</mi><mo>′</mo></msubsup><mo>=</mo><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msup><mrow><mo>(</mo><mfrac><msub><mi>L</mi><mtext>target</mtext></msub><msub><mi>L</mi><mtext>train</mtext></msub></mfrac><mo>)</mo></mrow><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo>/</mo><mi>d</mi></mrow></msup></mrow></mrow></semantics></math>Variants include:<span class="hh-tag">•</span>Linear scaling (Position Interpolation) [48]: Simply divide position indices by a factor <math alttext="s" display="inline"><semantics><mi>s</mi></semantics></math>. Cheap but degrades quality at high extension ratios.<span class="hh-tag">•</span>NTK-aware scaling[289]: Scale the base frequency <math alttext="\theta=10000\to 10000\cdot s^{d/(d-2)}" display="inline"><semantics><mrow><mi>θ</mi><mo>=</mo><mn>10000</mn><mo stretchy="false">→</mo><mrow><mn>10000</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msup><mi>s</mi><mrow><mi>d</mi><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mi>d</mi><mo>−</mo><mn>2</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mrow></semantics></math>. Preserves high-frequency (local) information while extending low-frequency (global) range.<span class="hh-tag">•</span>YaRN[289] (Yet another RoPE extensioN): Combines NTK scaling with an attention temperature correction and fine-tuning on a small long-context corpus. Used by Llama-3 to extend from 8K training to 128K deployment.<span class="hh-tag">•</span>Dynamic NTK[289]: Adjusts the scaling factor on-the-fly based on actual sequence length at inference. No fixed extension ratio needed—the model adapts as context grows.</li><li>Continued pretraining on long data: Even with RoPE scaling, models benefit from a short continued pretraining phase (1–5B tokens) on long documents. This teaches the model to actually <em>use</em> distant context, not just tolerate it positionally. Llama-3.1 used a progressive schedule: 8K <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 64K <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 128K.</li><li>Ring Attention / Blockwise Parallel[222]: For sequences exceeding single-GPU memory (1M+ tokens), Ring Attention distributes the sequence across GPUs in a ring topology. Each GPU holds a block and passes KV blocks around the ring, computing local attention tiles. This enables linear memory scaling with GPU count while preserving exact attention.</li><li>Hybrid architectures: Some systems combine a local sliding window (e.g., 4K) for most layers with full attention at select layers (e.g., every 4th layer). This provides <math alttext="O(n\cdot w)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>w</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math> cost for most computation while maintaining global information flow.</li></ol>

### Feed-Forward Network (MLP)

Each transformer block contains an MLP applied independently to each position:

<div class="hh-equation" id="ch1.ex13"><math alttext="\text{FFN}(x)=W_{2}\cdot\sigma(W_{1}x+b_{1})+b_{2}" display="block"><semantics><mrow><mrow><mtext>FFN</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><msub><mi>W</mi><mn>2</mn></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msub><mi>W</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow><mo>+</mo><msub><mi>b</mi><mn>1</mn></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>+</mo><msub><mi>b</mi><mn>2</mn></msub></mrow></mrow></semantics></math></div>

where <math alttext="W_{1}\in\mathbb{R}^{d\times 4d}" display="inline"><semantics><mrow><msub><mi>W</mi><mn>1</mn></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mi>d</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mn>4</mn></mrow><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow></msup></mrow></semantics></math>, <math alttext="W_{2}\in\mathbb{R}^{4d\times d}" display="inline"><semantics><mrow><msub><mi>W</mi><mn>2</mn></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mn>4</mn><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math>. Modern LLMs use:

<ul><li>SwiGLU activation: <math alttext="\text{FFN}(x)=W_{2}(\text{Swish}(W_{1}x)\odot W_{3}x)" display="inline"><semantics><mrow><mrow><mtext>FFN</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>W</mi><mn>2</mn></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mrow><mtext>Swish</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>W</mi><mn>1</mn></msub><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⊙</mo><msub><mi>W</mi><mn>3</mn></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mi>x</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> — used by Llama [118], Mistral [165]. Requires 3 weight matrices but gives better performance.</li><li>Hidden dimension is typically <math alttext="8/3\times d" display="inline"><semantics><mrow><mrow><mn>8</mn><mo>/</mo><mn>3</mn></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></semantics></math> (rounded to multiples of 256 for Tensor Core efficiency).</li></ul>

### Layer Normalization

Layer normalization stabilizes training by normalizing activations across the feature dimension. Its placement relative to the attention/FFN sublayers significantly affects training dynamics.

Given a hidden state vector <math alttext="\mathbf{x}\in\mathbb{R}^{d}" display="inline"><semantics><mrow><mi>𝐱</mi><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math> (a single token’s representation), LayerNorm [15] computes:

<div class="hh-equation" id="ch1.e5"><math alttext="\text{LayerNorm}(\mathbf{x})=\gamma\odot\frac{\mathbf{x}-\mu}{\sqrt{\sigma^{2}+\epsilon}}+\beta" display="block"><semantics><mrow><mrow><mtext>LayerNorm</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>γ</mi><mo lspace="0.222em" rspace="0.222em">⊙</mo><mfrac><mrow><mi>𝐱</mi><mo>−</mo><mi>μ</mi></mrow><msqrt><mrow><msup><mi>σ</mi><mn>2</mn></msup><mo>+</mo><mi>ϵ</mi></mrow></msqrt></mfrac></mrow><mo>+</mo><mi>β</mi></mrow></mrow></semantics></math><span class="hh-equation-number">(1.5)</span></div>

where:

<ul><li><math alttext="\mu=\frac{1}{d}\sum_{i=1}^{d}x_{i}" display="inline"><semantics><mrow><mi>μ</mi><mo>=</mo><mrow><mfrac><mn>1</mn><mi>d</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msubsup><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></msubsup><msub><mi>x</mi><mi>i</mi></msub></mrow></mrow></mrow></semantics></math> (mean across the <math alttext="d" display="inline"><semantics><mi>d</mi></semantics></math> feature dimensions)</li><li><math alttext="\sigma^{2}=\frac{1}{d}\sum_{i=1}^{d}(x_{i}-\mu)^{2}" display="inline"><semantics><mrow><msup><mi>σ</mi><mn>2</mn></msup><mo>=</mo><mrow><mfrac><mn>1</mn><mi>d</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msubsup><mo rspace="0em">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></msubsup><msup><mrow><mo stretchy="false">(</mo><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mi>μ</mi></mrow><mo stretchy="false">)</mo></mrow><mn>2</mn></msup></mrow></mrow></mrow></semantics></math> (variance across features)</li><li><math alttext="\gamma,\beta\in\mathbb{R}^{d}" display="inline"><semantics><mrow><mrow><mi>γ</mi><mo>,</mo><mi>β</mi></mrow><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math> are learned scale and shift parameters (per-dimension)</li><li><math alttext="\epsilon\approx 10^{-5}" display="inline"><semantics><mrow><mi>ϵ</mi><mo>≈</mo><msup><mn>10</mn><mrow><mo>−</mo><mn>5</mn></mrow></msup></mrow></semantics></math> prevents division by zero</li></ul>

Key distinction from BatchNorm: LayerNorm normalizes across the <em>feature dimension</em> of a single example, not across the batch. This makes it independent of batch size and works identically at training and inference.

RMSNorm [421] drops the mean-centering step, normalizing only by the root-mean-square:

<div class="hh-equation" id="ch1.e6"><math alttext="\text{RMSNorm}(\mathbf{x})=\gamma\odot\frac{\mathbf{x}}{\text{RMS}(\mathbf{x})},\qquad\text{RMS}(\mathbf{x})=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_{i}^{2}}" display="block"><semantics><mrow><mrow><mrow><mtext>RMSNorm</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>γ</mi><mo lspace="0.222em" rspace="0.222em">⊙</mo><mfrac><mi>𝐱</mi><mrow><mtext>RMS</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow></mfrac></mrow></mrow><mo rspace="2.167em">,</mo><mrow><mrow><mtext>RMS</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>𝐱</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><msqrt><mrow><mfrac><mn>1</mn><mi>d</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>d</mi></munderover><msubsup><mi>x</mi><mi>i</mi><mn>2</mn></msubsup></mrow></mrow></msqrt></mrow></mrow></semantics></math><span class="hh-equation-number">(1.6)</span></div>

No <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> (shift) parameter and no mean subtraction — just scale. This saves one reduction operation per token and is <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>5–10% faster on GPUs while achieving equivalent model quality. All modern LLMs (Llama, Mistral, Qwen) use RMSNorm.

### Model Size Reference

The following table summarizes key architectural parameters for widely-used open-weight models (latest versions as of 2025), providing a quick reference for understanding scale and design choices.

<div class="hh-table" id="ch1.t7"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Model</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Params</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Layers</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="d" display="inline"><semantics><mi>d</mi><span></span></semantics></math></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Heads</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>KV Heads</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Context</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Llama-3.1 8B <span>[118]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8B</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>32</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4096</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>32</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128K</span></span></td></tr><tr><th class="ltx_align_left">Llama-3.1 405B <span>[118]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>405B</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>126</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>16384</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128K</span></span></td></tr><tr><th class="ltx_align_left">Llama-4 Maverick <span>[4]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>400B (17B active)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>48</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>5120</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>40</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1M</span></span></td></tr><tr><th class="ltx_align_left">Mistral Large 2 <span>[5]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>123B</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>88</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>12288</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>96</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128K</span></span></td></tr><tr><th class="ltx_align_left">Qwen-2.5 72B <span>[350]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>72B</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>80</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8192</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>64</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128K</span></span></td></tr><tr><th class="ltx_align_left">DeepSeek-V3 <span>[71]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>671B (37B active)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>61</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>7168</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>MLA</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>128K</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.7:</span>Architecture parameters for popular open-weight LLMs (2024–2025 generation).</p></div>

<em>Note</em>: Models marked with “active” parameters use Mixture-of-Experts (MoE) architecture—total parameters indicate model capacity, while active parameters reflect per-token compute cost. DeepSeek-V3 uses Multi-head Latent Attention (MLA) instead of standard GQA, compressing KV into a low-rank latent space.

### Attention Pathologies

While the attention mechanism is powerful, it exhibits systematic failure modes that practitioners must understand—especially when scaling to long contexts or interpreting model behaviour.

Xiao et al. [393] discovered that transformer models allocate disproportionately high attention scores to the <em>first token</em> in the sequence—regardless of its semantic content. Even when the first token is a meaningless <code>⟨BOS⟩</code> marker, attention heads across all layers consistently attend to it, sometimes with 20–50% of total attention mass.

Softmax attention must produce a valid probability distribution (<math alttext="\sum_{j}\alpha_{j}=1" display="inline"><semantics><mrow><mrow><msub><mo>∑</mo><mi>j</mi></msub><msub><mi>α</mi><mi>j</mi></msub></mrow><mo>=</mo><mn>1</mn></mrow></semantics></math>). When no key is particularly relevant to a query, the model needs a “dump” location for unused attention mass. During training, the first token becomes this default sink because it is always present and positionally predictable. It functions as a <em>no-op attention target</em>—the model has learned to route irrelevant attention there rather than distributing it unpredictably.

<div class="hh-equation" id="ch1.ex14"><math alttext="\alpha_{\text{sink}}=\frac{\exp(q^{\top}k_{0}/\sqrt{d})}{\sum_{j}\exp(q^{\top}k_{j}/\sqrt{d})}\gg\frac{1}{n}\quad\text{(even when }k_{0}\text{ is semantically irrelevant)}" display="block"><semantics><mrow><mrow><msub><mi>α</mi><mtext>sink</mtext></msub><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msup><mi>q</mi><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>k</mi><mn>0</mn></msub></mrow><mo>/</mo><msqrt><mi>d</mi></msqrt></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mo>∑</mo><mi>j</mi></msub><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msup><mi>q</mi><mo>⊤</mo></msup><mo lspace="0em" rspace="0em">​</mo><msub><mi>k</mi><mi>j</mi></msub></mrow><mo>/</mo><msqrt><mi>d</mi></msqrt></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac><mo>≫</mo><mfrac><mn>1</mn><mi>n</mi></mfrac></mrow><mspace width="1em"></mspace><mrow><mtext>(even when </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>k</mi><mn>0</mn></msub><mo lspace="0em" rspace="0em">​</mo><mtext> is semantically irrelevant)</mtext></mrow></mrow></semantics></math></div>

<ul><li>Streaming inference failure: When using sliding-window KV caches, evicting the first token causes perplexity to spike catastrophically—the model loses its attention sink.</li><li>Misleading interpretability: Naive attention visualizations suggest the first token is “important” when it is merely a mathematical artefact.</li><li>Context window waste: The sink token occupies a KV cache slot without carrying useful information.</li></ul>

<ul><li>StreamingLLM[393]: Always keep the first <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> tokens (“attention sinks”) in the KV cache alongside the recent sliding window. Enables infinite-length generation with bounded memory.</li><li>Sink tokens by design: Some models (e.g., Mistral) prepend dedicated sink tokens during training that are explicitly meant to absorb residual attention.</li><li>Softmax alternatives: Replace softmax with ReLU attention or sigmoid gating, where zero attention is representable without requiring a dump target.</li></ul>

As sequence length <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> grows, each query must distribute its attention budget across more keys. The average attention weight per token decreases as <math alttext="O(1/n)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>/</mo><mi>n</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math>, making it progressively harder for the model to concentrate on the few truly relevant positions—a problem known as <em>attention dilution</em> or <em>attention diffusion</em>[227].

Liu et al. [227] showed that LLMs exhibit a U-shaped retrieval curve: information placed at the <em>beginning</em> or <em>end</em> of long contexts is retrieved reliably, but information in the <em>middle</em> is often ignored. This is a direct consequence of attention dilution compounded with positional biases from RoPE/ALiBi:

<ul><li>Softmax saturation: With many keys, the softmax temperature effectively decreases, making the distribution more uniform (entropic).</li><li>Positional decay: RoPE’s relative positional encoding introduces a natural decay with distance, suppressing attention to middle positions that are far from both start and end.</li><li>Training distribution: Models trained on shorter sequences develop attention patterns biased toward recent context.</li></ul>

<ul><li>Explicit retrieval: Place relevant context at the beginning or end of the prompt; use RAG to avoid relying on middle positions.</li><li>Long-context training: Train on long documents with varied placement of key information [100].</li><li>Hierarchical attention: Architectures like Mamba [123] or RWKV that avoid the <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> attention bottleneck entirely.</li><li>Landmark tokens: Insert retrievable markers in the context that act as “signposts” for attention.</li><li>Temperature scaling: Some implementations scale the attention logits by <math alttext="\log n" display="inline"><semantics><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mi>n</mi></mrow></semantics></math> to counteract dilution in long sequences.</li></ul>

<div class="hh-table" id="ch1.t8"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Pattern</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Description</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Implication</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Attention heads specialization</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Different heads learn distinct roles: syntax heads, co-reference heads, positional heads <span>[361]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Not all heads are equally important; many can be pruned</span></span></td></tr><tr><th class="ltx_align_left"><span>Induction heads</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Heads that implement [A][B]…[A] <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math> [B] copying <span>[273]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Critical for in-context learning; emerge in 2-layer+ models</span></span></td></tr><tr><th class="ltx_align_left"><span>Attention collapse</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>In deep networks, attention distributions can converge (all heads attend same positions)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Hurts expressivity; addressed by attention diversity losses</span></span></td></tr><tr><th class="ltx_align_left"><span>Retrieval heads</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Specific heads specialize in retrieving factual information from context <span>[387]</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Explains why pruning certain heads causes hallucination spikes</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.8:</span>Additional attention patterns observed in large transformers.</p></div>

### Visualizing Attention for Explainability

Attention weights provide a window into model reasoning—but must be interpreted carefully.

The simplest approach: plot the <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> attention matrix <math alttext="A=\text{softmax}(QK^{\top}/\sqrt{d})" display="inline"><semantics><mrow><mi>A</mi><mo>=</mo><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mo>⊤</mo></msup></mrow><mo>/</mo><msqrt><mi>d</mi></msqrt></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> as a heatmap for each head and layer. Tools like BertViz [359] render interactive multi-head visualizations.

Raw attention at a single layer is misleading because information flows through residual connections across <em>all</em> layers. Abnar and Zuidema [2] propose <em>attention rollout</em>: multiply attention matrices across layers to approximate the total information flow from input to output:

<div class="hh-equation" id="ch1.ex15"><math alttext="R^{(l)}=A^{(l)}\cdot R^{(l-1)},\quad R^{(0)}=I" display="block"><semantics><mrow><mrow><msup><mi>R</mi><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><msup><mi>A</mi><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><msup><mi>R</mi><mrow><mo stretchy="false">(</mo><mrow><mi>l</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></msup></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msup><mi>R</mi><mrow><mo stretchy="false">(</mo><mn>0</mn><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mi>I</mi></mrow></mrow></semantics></math></div>

where <math alttext="A^{(l)}" display="inline"><semantics><msup><mi>A</mi><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup></semantics></math> is the (averaged across heads) attention matrix at layer <math alttext="l" display="inline"><semantics><mi>l</mi></semantics></math>, adjusted to include the residual connection: <math alttext="A^{(l)}=0.5\cdot A^{(l)}_{\text{raw}}+0.5\cdot I" display="inline"><semantics><mrow><msup><mi>A</mi><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msup><mo>=</mo><mrow><mrow><mn>0.5</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msubsup><mi>A</mi><mtext>raw</mtext><mrow><mo stretchy="false">(</mo><mi>l</mi><mo stretchy="false">)</mo></mrow></msubsup></mrow><mo>+</mo><mrow><mn>0.5</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>I</mi></mrow></mrow></mrow></semantics></math>.

Combine attention weights with gradient information to identify which attended tokens actually <em>influence</em> the output [20]:

<div class="hh-equation" id="ch1.ex16"><math alttext="\text{Relevance}(i)=\alpha_{i}\cdot\left|\frac{\partial y}{\partial h_{i}}\right|" display="block"><semantics><mrow><mrow><mtext>Relevance</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>i</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><msub><mi>α</mi><mi>i</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo>|</mo><mfrac><mrow><mo>∂</mo><mi>y</mi></mrow><mrow><mo>∂</mo><msub><mi>h</mi><mi>i</mi></msub></mrow></mfrac><mo>|</mo></mrow></mrow></mrow></semantics></math></div>

This addresses the criticism that high attention <math alttext="\neq" display="inline"><semantics><mo>≠</mo></semantics></math> high influence (a token can receive high attention but be processed through a near-zero-weight path).

Individual neurons in transformer MLPs and residual streams are typically <em>polysemantic</em>—a single neuron activates for multiple unrelated concepts (e.g., “the colour blue AND academic citations AND the word ‘the’”). This makes direct neuron-level interpretation unreliable.

Cunningham et al. [63] and Bricken et al. [30] demonstrated that training a sparse autoencoder (SAE) on model activations can decompose polysemantic representations into <em>monosemantic features</em>—interpretable directions that each correspond to a single concept:

<div class="hh-equation" id="ch1.ex17"><math alttext="h=W_{\text{dec}}\cdot\text{ReLU}(W_{\text{enc}}\cdot x+b_{\text{enc}})+b_{\text{dec}}" display="block"><semantics><mrow><mi>h</mi><mo>=</mo><mrow><mrow><mrow><msub><mi>W</mi><mtext>dec</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>ReLU</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msub><mi>W</mi><mtext>enc</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>x</mi></mrow><mo>+</mo><msub><mi>b</mi><mtext>enc</mtext></msub></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><msub><mi>b</mi><mtext>dec</mtext></msub></mrow></mrow></semantics></math></div>

where <math alttext="W_{\text{enc}}\in\mathbb{R}^{m\times d}" display="inline"><semantics><mrow><msub><mi>W</mi><mtext>enc</mtext></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>m</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math> with <math alttext="m\gg d" display="inline"><semantics><mrow><mi>m</mi><mo>≫</mo><mi>d</mi></mrow></semantics></math> (overcomplete basis), and the ReLU + sparsity penalty ensures only a few features activate per input.

<ul><li>Features are <em>monosemantic</em>: each encodes a single human-interpretable concept (“code in Python,” “mentions of the Golden Gate Bridge,” “first-person narrative”) [30].</li><li>Features are <em>steerable</em>: clamping a feature’s activation high/low directly controls model behaviour (e.g., forcing the “Golden Gate Bridge” feature on makes the model mention it in every response) [353].</li><li>Features compose: complex behaviours emerge from combinations of simple features.</li><li>SAEs scale: Templeton et al. [353] trained SAEs with up to 34M features on Claude 3 Sonnet, finding interpretable features for safety-relevant concepts (deception, sycophancy, dangerous requests).</li></ul>

While SAEs decompose activations into interpretable <em>vectors</em>, their features still require human inspection of max-activating examples to understand. Anthropic’s Natural Language Autoencoders (NLAEs) [12] take a fundamentally different approach: they replace the sparse bottleneck with <em>natural language descriptions</em>, making interpretability automatic.

<ol><li>Encoder: A language model reads the hidden activations (or the input text) and produces a natural language description of the active concepts: e.g., “The text discusses French cuisine and uses formal academic tone.”</li><li>Decoder: A second language model reads the natural language description and reconstructs the original activations (or predicts the next token).</li><li>Training: Both encoder and decoder are trained end-to-end to minimize reconstruction loss, with the bottleneck being a variable-length natural language string rather than a sparse vector.</li></ol>

<ul><li>Self-interpreting: Features are <em>literally</em> natural language—no manual labelling needed.</li><li>Compositional: Can express complex, relational concepts (“a sarcastic response to a factual claim”) that SAE features cannot represent as single directions.</li><li>Hierarchical: Descriptions can capture both fine-grained (word-level) and coarse (document-level) properties in the same representation.</li><li>Auditable: The bottleneck description is human-readable, enabling direct inspection of what information the model “thinks” is present.</li></ul>

NLAEs introduce a language-model-in-the-loop, making them computationally expensive and potentially subject to the same faithfulness concerns as any model-generated explanation. They also cannot easily represent sub-symbolic features (geometric patterns, exact numerical values) that SAEs handle naturally as activation magnitudes.

## Prediction Heads: What Transformers Output

The transformer body produces contextual hidden states <math alttext="\mathbf{h}_{t}\in\mathbb{R}^{d}" display="inline"><semantics><mrow><msub><mi>𝐡</mi><mi>t</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math> for each position. What we <em>do</em> with these hidden states—the prediction head—defines the task. The same transformer backbone can serve radically different purposes simply by swapping the head.

<figure id="ch1.f7"><img src="./fig_008_prediction-heads.png" alt="Figure 1.7: The same transformer backbone supports different tasks by swapping the prediction head. All three heads used" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.7:</span>The same transformer backbone supports different tasks by swapping the prediction head. All three heads used in this paper share identical architecture below the final projection layer.</figcaption></figure>

### Language Modeling Head (Pretraining)

The standard LM head projects the final hidden state to vocabulary logits and trains with cross-entropy loss over the next token:

<div class="hh-equation" id="ch1.e7"><math alttext="P(x_{t+1}|x_{\leq t})=\text{softmax}(\mathbf{W}_{\text{head}}\cdot\mathbf{h}_{t}+\mathbf{b})" display="block"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>≤</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><msub><mi>𝐖</mi><mtext>head</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>𝐡</mi><mi>t</mi></msub></mrow><mo>+</mo><mi>𝐛</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(1.7)</span></div>

where <math alttext="\mathbf{W}_{\text{head}}\in\mathbb{R}^{|\mathcal{V}|\times d}" display="inline"><semantics><mrow><msub><mi>𝐖</mi><mtext>head</mtext></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo rspace="0.055em" stretchy="false">|</mo></mrow><mo rspace="0.222em">×</mo><mi>d</mi></mrow></msup></mrow></semantics></math> (often tied with the embedding matrix: <math alttext="\mathbf{W}_{\text{head}}=\mathbf{E}^{T}" display="inline"><semantics><mrow><msub><mi>𝐖</mi><mtext>head</mtext></msub><mo>=</mo><msup><mi>𝐄</mi><mi>T</mi></msup></mrow></semantics></math>).

### Conditional Generation Head (SFT / Instruction Following)

For supervised fine-tuning (SFT), the architecture is <em>identical</em> to the LM head—the same linear projection to vocabulary logits. The difference is purely in <em>what we compute loss on</em>:

<div class="hh-equation" id="ch1.e8"><math alttext="\mathcal{L}_{\text{SFT}}=-\frac{1}{|y|}\sum_{t=1}^{|y|}\log P(y_{t}|x_{\text{prompt}},y_{&lt;t})" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>SFT</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></munderover><mi>log</mi><mi>P</mi><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo fence="false" rspace="0.167em" stretchy="false">|</mo><msub><mi>x</mi><mtext>prompt</mtext></msub><mo>,</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(1.8)</span></div>

### Value Head (Regression for RL)

In reinforcement learning (PPO, GRPO), we need to estimate <em>how good</em> a state is—this requires a scalar output, not vocabulary logits. The value head replaces the LM projection with a simple regression layer:

<div class="hh-equation" id="ch1.e9"><math alttext="V(s_{t})=\mathbf{w}_{\text{value}}^{T}\cdot\mathbf{h}_{t}+b\in\mathbb{R}" display="block"><semantics><mrow><mrow><mi>V</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>s</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><msubsup><mi>𝐰</mi><mtext>value</mtext><mi>T</mi></msubsup><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>𝐡</mi><mi>t</mi></msub></mrow><mo>+</mo><mi>b</mi></mrow><mo>∈</mo><mi>ℝ</mi></mrow></semantics></math><span class="hh-equation-number">(1.9)</span></div>

where <math alttext="\mathbf{w}_{\text{value}}\in\mathbb{R}^{d}" display="inline"><semantics><mrow><msub><mi>𝐰</mi><mtext>value</mtext></msub><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math> and <math alttext="b\in\mathbb{R}" display="inline"><semantics><mrow><mi>b</mi><mo>∈</mo><mi>ℝ</mi></mrow></semantics></math>.

### Head Selection Summary

<div class="hh-table" id="ch1.t9"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Head</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Output</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Loss</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Stage</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Purpose</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">LM Head</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\mathbb{R}^{|\mathcal{V}|}" display="inline"><semantics><msup><mi>ℝ</mi><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow></msup><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cross-entropy (all tokens)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Pretraining</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learn language from raw text</span></span></td></tr><tr><th class="ltx_align_left">Conditional Head</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\mathbb{R}^{|\mathcal{V}|}" display="inline"><semantics><msup><mi>ℝ</mi><mrow><mo stretchy="false">|</mo><mi>𝒱</mi><mo stretchy="false">|</mo></mrow></msup><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cross-entropy (response only)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>SFT</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learn to follow instructions</span></span></td></tr><tr><th class="ltx_align_left">Value Head</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\mathbb{R}^{1}" display="inline"><semantics><msup><mi>ℝ</mi><mn>1</mn></msup><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>MSE</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>RL (PPO)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Estimate state value for advantage</span></span></td></tr><tr><th class="ltx_align_left">Reward Head</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\mathbb{R}^{1}" display="inline"><semantics><msup><mi>ℝ</mi><mn>1</mn></msup><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Pairwise ranking</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>RM training</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Score response quality</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.9:</span>Prediction heads used throughout this paper and their training contexts.</p></div>

### HuggingFace Implementation

```
from transformers import (
    AutoModelForCausalLM,          # LM head (pretraining + SFT)
    AutoModelForSequenceClassification,  # Reward head
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead  # Value head (PPO)
import torch

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# === 1. LM Head (Pretraining / SFT) ===
# The default CausalLM model -- projects hidden states to vocab logits
lm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# lm_model.lm_head: Linear(hidden_size -> vocab_size)
# Output: logits of shape (batch, seq_len, vocab_size)

inputs = tokenizer("The capital of France is", return_tensors="pt")
outputs = lm_model(**inputs)
next_token_logits = outputs.logits[:, -1, :]  # (batch, vocab_size)
probs = torch.softmax(next_token_logits, dim=-1)

# === 2. Conditional Head (SFT) ===
# Architecturally identical to LM head -- difference is in loss masking
# During SFT, we only compute loss on response tokens:
messages = [
    {"role": "user", "content": "What is 2+2?"},
    {"role": "assistant", "content": "4"},
]
formatted = tokenizer.apply_chat_template(messages, return_tensors="pt")
labels = formatted.clone()
# Mask prompt tokens (set to -100 so cross-entropy ignores them)
prompt_len = len(tokenizer.apply_chat_template(messages[:1]))
labels[:, :prompt_len] = -100
loss = lm_model(input_ids=formatted, labels=labels).loss

# === 3. Value Head (PPO Critic) ===
# Adds a Linear(hidden_size -> 1) on top of the LM backbone
value_model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# value_model.v_head: Linear(hidden_size -> 1)
# Returns both LM logits AND per-token value estimates

inputs = tokenizer("Explain quantum computing", return_tensors="pt")
lm_logits, loss, values = value_model(
    **inputs, return_dict=False
)
# values shape: (batch, seq_len, 1) -- scalar estimate per token

# === 4. Reward Head (Reward Model) ===
# Classification head: Linear(hidden_size -> 1) on last token
reward_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,              # single scalar output
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
# Scores entire sequence by pooling the last token's hidden state
inputs = tokenizer("Good response here", return_tensors="pt")
reward_score = reward_model(**inputs).logits  # shape: (batch, 1)
```

<span class="hh-tag">Listing 3:</span>Loading and using different prediction heads with HuggingFace.

## Optimization Theory for LLM Training

Training a large language model means finding the set of parameters <math alttext="\theta" display="inline"><semantics><mi>θ</mi></semantics></math> (billions of weights) that minimizes the loss function <math alttext="\mathcal{L}(\theta)" display="inline"><semantics><mrow><mi>ℒ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> — typically the negative log-likelihood of the next token. This is an optimization problem in extraordinarily high-dimensional space, and the algorithm used to navigate this space determines whether training succeeds, diverges, or stalls.

### Gradient Descent: The Foundation

The gradient <math alttext="\nabla_{\theta}\mathcal{L}" display="inline"><semantics><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>ℒ</mi></mrow></semantics></math> is a vector that points in the direction of <em>steepest increase</em> of the loss. Each component <math alttext="\frac{\partial\mathcal{L}}{\partial\theta_{i}}" display="inline"><semantics><mfrac><mrow><mo>∂</mo><mi>ℒ</mi></mrow><mrow><mo>∂</mo><msub><mi>θ</mi><mi>i</mi></msub></mrow></mfrac></semantics></math> tells us how much the loss would change if we slightly increased parameter <math alttext="\theta_{i}" display="inline"><semantics><msub><mi>θ</mi><mi>i</mi></msub></semantics></math>. To <em>decrease</em> the loss, we move in the opposite direction:

<div class="hh-equation" id="ch1.e10"><math alttext="\theta_{t+1}=\theta_{t}-\eta\nabla_{\theta}\mathcal{L}(\theta_{t})" display="block"><semantics><mrow><msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><msub><mi>θ</mi><mi>t</mi></msub><mo>−</mo><mrow><mi>η</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><mi>ℒ</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>θ</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(1.10)</span></div>

where <math alttext="\eta&gt;0" display="inline"><semantics><mrow><mi>η</mi><mo>&gt;</mo><mn>0</mn></mrow></semantics></math> is the learning rate — the step size. This is gradient descent[310].

<figure id="ch1.f8"><img src="./fig_009_fig9.png" alt="Figure 1.8: Gradient descent: starting from a random initialization θ0\theta_{0}, each step moves the parameters in the " loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.8:</span>Gradient descent: starting from a random initialization <math alttext="\theta_{0}" display="inline"><semantics><msub><mi>θ</mi><mn>0</mn></msub></semantics></math>, each step moves the parameters in the direction that reduces the loss, with step size controlled by the learning rate <math alttext="\eta" display="inline"><semantics><mi>η</mi></semantics></math>. The process converges toward a (local) minimum.</figcaption></figure>

Computing the exact gradient requires evaluating the loss over the <em>entire</em> training dataset (trillions of tokens for LLMs). This is computationally prohibitive — a single gradient step would require a full pass over all data.

The solution: estimate the gradient from a small random subset (mini-batch) of the data [308]:

<div class="hh-equation" id="ch1.ex18"><math alttext="\nabla_{\theta}\mathcal{L}(\theta)\approx\frac{1}{B}\sum_{i=1}^{B}\nabla_{\theta}\ell(\theta;x_{i})" display="block"><semantics><mrow><mrow><mrow><msub><mo>∇</mo><mi>θ</mi></msub><mi>ℒ</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo stretchy="false">)</mo></mrow></mrow><mo>≈</mo><mrow><mfrac><mn>1</mn><mi>B</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><mrow><mrow><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><mi mathvariant="normal">ℓ</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>θ</mi><mo>,</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math></div>

where <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> is the batch size (typically 1K–4M tokens for LLMs). The mini-batch gradient is a <em>noisy but unbiased</em> estimate of the true gradient.

While SGD with momentum works well for vision models (CNNs), LLM training requires adaptive optimizers — algorithms that maintain a per-parameter learning rate.

### Why Vanilla SGD Fails for LLMs

Stochastic Gradient Descent updates weights as:

<div class="hh-equation" id="ch1.ex19"><math alttext="\theta_{t+1}=\theta_{t}-\eta\nabla_{\theta}\mathcal{L}(\theta_{t})" display="block"><semantics><mrow><msub><mi>θ</mi><mrow><mi>t</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><msub><mi>θ</mi><mi>t</mi></msub><mo>−</mo><mrow><mi>η</mi><mo lspace="0.167em" rspace="0em">​</mo><mrow><msub><mo rspace="0.167em">∇</mo><mi>θ</mi></msub><mi>ℒ</mi></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>θ</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math></div>

### Adam – Adaptive Moment Estimation

Adam [185] maintains per-parameter estimates of the first moment (mean of gradients) and second moment (uncentered variance of gradients).

### AdamW – Decoupled Weight Decay

AdamW [239] fixes a subtle but important issue with how weight decay interacts with adaptive optimizers.

### Muon: Beyond AdamW

For nearly a decade, AdamW was the unchallenged default optimizer for neural network training. Muon [225] (Liu et al., 2025) is the first serious challenger to that dominance. Its core insight is that the momentum buffer accumulated by Adam carries directional information that is obscured by the adaptive scaling step—and that <em>orthogonalizing</em> that momentum before applying it produces a fundamentally better update direction.

The Newton-Schulz iteration is a matrix polynomial that converges to the orthogonal factor of the polar decomposition of <math alttext="M_{t}" display="inline"><semantics><msub><mi>M</mi><mi>t</mi></msub></semantics></math>. Concretely, starting from <math alttext="X_{0}=M_{t}/\|M_{t}\|_{F}" display="inline"><semantics><mrow><msub><mi>X</mi><mn>0</mn></msub><mo>=</mo><mrow><msub><mi>M</mi><mi>t</mi></msub><mo>/</mo><msub><mrow><mo stretchy="false">‖</mo><msub><mi>M</mi><mi>t</mi></msub><mo stretchy="false">‖</mo></mrow><mi>F</mi></msub></mrow></mrow></semantics></math>, the iteration

<div class="hh-equation" id="ch1.ex26"><math alttext="X_{k+1}=\frac{3}{2}X_{k}-\frac{1}{2}X_{k}X_{k}^{\top}X_{k}" display="block"><semantics><mrow><msub><mi>X</mi><mrow><mi>k</mi><mo>+</mo><mn>1</mn></mrow></msub><mo>=</mo><mrow><mrow><mfrac><mn>3</mn><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><msub><mi>X</mi><mi>k</mi></msub></mrow><mo>−</mo><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><msub><mi>X</mi><mi>k</mi></msub><mo lspace="0em" rspace="0em">​</mo><msubsup><mi>X</mi><mi>k</mi><mo>⊤</mo></msubsup><mo lspace="0em" rspace="0em">​</mo><msub><mi>X</mi><mi>k</mi></msub></mrow></mrow></mrow></semantics></math></div>

converges rapidly (typically 5–10 steps) to a matrix <math alttext="\tilde{M}_{t}" display="inline"><semantics><msub><mover accent="true"><mi>M</mi><mo>~</mo></mover><mi>t</mi></msub></semantics></math> satisfying <math alttext="\tilde{M}_{t}^{\top}\tilde{M}_{t}\approx I" display="inline"><semantics><mrow><mrow><msubsup><mover accent="true"><mi>M</mi><mo>~</mo></mover><mi>t</mi><mo>⊤</mo></msubsup><mo lspace="0em" rspace="0em">​</mo><msub><mover accent="true"><mi>M</mi><mo>~</mo></mover><mi>t</mi></msub></mrow><mo>≈</mo><mi>I</mi></mrow></semantics></math>. Orthogonality ensures that the update moves weight matrices in a direction that is maximally spread across all singular value directions—preventing the optimizer from collapsing updates onto a low-rank subspace, which is a known failure mode of Adam on weight matrices with highly skewed singular value spectra.

Muon claims approximately 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> compute efficiency relative to AdamW: the same validation loss is reached in roughly half the gradient steps. This is not a marginal improvement—it represents a qualitative shift in the optimizer’s sample efficiency on the loss landscape.

The momentum is clearly shifting toward Muon in large-scale pretraining:

<ul><li>GLM-4.5 / GLM-5 (Zhipu AI) [434, 435]: adopted Muon for pretraining</li><li>Kimi K2 (Moonshot AI) [17]: uses MuonClip, a variant that adds <em>QK-Clip</em>—rescaling query and key projections to cap attention logits and prevent loss spikes. Kimi K2 trained over 15.5T tokens with zero loss spikes, a stability record for models at that scale.</li><li>DeepSeek-V4 (2026) [72]: adopted Muon for its pretraining run</li></ul>

### Learning Rate – The Most Important Hyperparameter

### Learning Rate Warmup

<ul><li>Linear warmup:<math alttext="\eta_{t}=\eta_{\max}\times t/T_{\text{warmup}}" display="inline"><semantics><mrow><msub><mi>η</mi><mi>t</mi></msub><mo>=</mo><mrow><mrow><msub><mi>η</mi><mi>max</mi></msub><mo lspace="0.222em" rspace="0.222em">×</mo><mi>t</mi></mrow><mo>/</mo><msub><mi>T</mi><mtext>warmup</mtext></msub></mrow></mrow></semantics></math></li><li>Typical warmup duration: 1–5% of total steps for pretraining; 3–10% for fine-tuning (shorter runs need proportionally more warmup)</li><li>For SFT: 50–200 warmup steps is typical</li></ul>

### Learning Rate Schedules

<figure id="ch1.f9"><img src="./fig_010_fig10.png" alt="Figure 1.9: Common learning rate schedules. All include a linear warmup phase. WSD (Warmup-Stable-Decay) is the emerging" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.9:</span>Common learning rate schedules. All include a linear warmup phase. WSD (Warmup-Stable-Decay) is the emerging standard for pretraining.</figcaption></figure>

Simplest schedule. Good for short fine-tuning runs where you want to avoid over-decaying the LR. Risk: no annealing means the model may not converge to the sharpest minimum.

<div class="hh-equation" id="ch1.ex27"><math alttext="\eta_{t}=\eta_{\min}+\frac{1}{2}(\eta_{\max}-\eta_{\min})\left(1+\cos\!\left(\frac{t-T_{\text{warmup}}}{T-T_{\text{warmup}}}\pi\right)\right)" display="block"><semantics><mrow><msub><mi>η</mi><mi>t</mi></msub><mo>=</mo><mrow><msub><mi>η</mi><mi>min</mi></msub><mo>+</mo><mrow><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>η</mi><mi>max</mi></msub><mo>−</mo><msub><mi>η</mi><mi>min</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mrow><mn>1</mn><mo>+</mo><mrow><mpadded width="1.216em"><mi>cos</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mfrac><mrow><mi>t</mi><mo>−</mo><msub><mi>T</mi><mtext>warmup</mtext></msub></mrow><mrow><mi>T</mi><mo>−</mo><msub><mi>T</mi><mtext>warmup</mtext></msub></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mi>π</mi></mrow><mo>)</mo></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow></mrow></semantics></math></div>

Standard for pretraining and SFT. Smooth decay avoids abrupt LR changes. <math alttext="\eta_{\min}" display="inline"><semantics><msub><mi>η</mi><mi>min</mi></msub></semantics></math> is typically <math alttext="\eta_{\max}/10" display="inline"><semantics><mrow><msub><mi>η</mi><mi>max</mi></msub><mo>/</mo><mn>10</mn></mrow></semantics></math>.

Simpler than cosine, similar empirical results. Preferred when you want predictable LR at any step.

The new standard for large-scale pretraining [150, 118]. Three phases:

<ol><li>Warmup: Linear ramp to <math alttext="\eta_{\max}" display="inline"><semantics><msub><mi>η</mi><mi>max</mi></msub></semantics></math> (1–5% of steps)</li><li>Stable: Constant <math alttext="\eta_{\max}" display="inline"><semantics><msub><mi>η</mi><mi>max</mi></msub></semantics></math> for the majority of training</li><li>Decay: Fast cosine or linear decay to <math alttext="\eta_{\min}" display="inline"><semantics><msub><mi>η</mi><mi>min</mi></msub></semantics></math> (last 10–20% of steps)</li></ol>

Key advantage: the stable phase allows checkpointing at any point and continuing training. The decay phase can be applied at the end of any run.

Periodic restarts reset the LR to <math alttext="\eta_{\max}" display="inline"><semantics><msub><mi>η</mi><mi>max</mi></msub></semantics></math>. Can help escape local minima. Less common for LLMs; more useful for smaller models.

### Gradient Clipping

The following snippet shows how the concepts from this section—AdamW with decoupled weight decay (§1.6.6), cosine learning-rate scheduling with linear warmup (§1.6.7), and gradient clipping (§1.6.8)—come together in practice using the HuggingFace <code>transformers</code> library.

```
from transformers import TrainingArguments, Trainer
from transformers import get_cosine_schedule_with_warmup
import torch

# --- Option 1: Using TrainingArguments (recommended) ---
training_args = TrainingArguments(
    output_dir="./checkpoints",

    # AdamW optimizer (decoupled weight decay, S1.6.6)
    optim="adamw_torch",
    learning_rate=2e-5,           # peak LR after warmup
    adam_beta1=0.9,               # first moment decay
    adam_beta2=0.999,             # second moment decay
    adam_epsilon=1e-8,            # numerical stability
    weight_decay=0.01,           # decoupled L2 penalty

    # Learning rate schedule (S1.6.7)
    lr_scheduler_type="cosine",  # cosine decay to 0
    warmup_ratio=0.1,            # 10% of steps = linear warmup

    # Gradient clipping (S1.6.8)
    max_grad_norm=1.0,           # clip by global L2 norm

    # Mixed precision (S1.6.9)
    bf16=True,                   # use BFloat16 on Ampere+ GPUs

    # Training duration
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # effective batch = 8*4 = 32
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

# --- Option 2: Manual control (for custom training loops) ---
from torch.optim import AdamW

# Separate weight-decay groups (don't regularize biases/norms)
no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
param_groups = [
    {
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": 0.01,
    },
    {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]

optimizer = AdamW(param_groups, lr=2e-5, betas=(0.9, 0.999))

# Cosine schedule with linear warmup
total_steps = len(train_dataloader) * num_epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

# Training loop with gradient clipping
for batch in train_dataloader:
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Clip gradients before optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

<span class="hh-tag">Listing 4:</span>Complete optimizer configuration combining AdamW – cosine schedule – and gradient clipping.

### Mixed Precision Training

<ol><li>Multiply loss by scale factor <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math> (e.g., <math alttext="S=2^{15}" display="inline"><semantics><mrow><mi>S</mi><mo>=</mo><msup><mn>2</mn><mn>15</mn></msup></mrow></semantics></math>)</li><li>Compute gradients in FP16 (scaled by <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math>)</li><li>Before optimizer step, divide gradients by <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math></li><li>Check for overflow (NaN/Inf); if found, skip step and reduce <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math></li><li>If no overflow for <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> consecutive steps, increase <math alttext="S" display="inline"><semantics><mi>S</mi></semantics></math></li></ol>

In mixed precision training, weights are stored in FP32 (master copy) and cast to BF16/FP16 for the forward/backward pass. The optimizer step is done in FP32. This is important because:

<ul><li>Small gradient updates (<math alttext="\Delta\theta\ll\theta" display="inline"><semantics><mrow><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>θ</mi></mrow><mo>≪</mo><mi>θ</mi></mrow></semantics></math>) would be lost in BF16 precision (7 mantissa bits <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 0.8% relative precision)</li><li>FP32 master weights ensure accurate accumulation of small updates over many steps</li><li>Memory cost: 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> weight storage (FP32 + BF16 copy)</li></ul>

```
# === HuggingFace TrainingArguments (simplest approach) ===
from transformers import TrainingArguments

# BF16 on Ampere+ GPUs (A100, H100, RTX 30xx/40xx)
args_bf16 = TrainingArguments(
    output_dir="./out",
    bf16=True,               # BF16 forward/backward; FP32 master weights
    bf16_full_eval=True,     # also use BF16 during evaluation
    # No loss scaling needed -- BF16 has FP32-equivalent range
)

# FP16 on older GPUs (V100, T4, RTX 20xx)
args_fp16 = TrainingArguments(
    output_dir="./out",
    fp16=True,               # FP16 forward/backward
    fp16_full_eval=False,    # keep eval in FP32 for accuracy
    # Loss scaling is automatic via PyTorch GradScaler
)

# === Manual PyTorch AMP (for custom training loops) ===
import torch

# Setup (PyTorch 2.x API)
use_fp16 = not torch.cuda.is_bf16_supported()
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)  # only needed for FP16
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
dtype = torch.float16 if use_fp16 else torch.bfloat16

for batch in train_dataloader:
    optimizer.zero_grad()

    # Autocast: run forward pass in reduced precision
    with torch.autocast("cuda", dtype=dtype):
        outputs = model(**batch)
        loss = outputs.loss

    if use_fp16:
        # FP16 path: scale loss to prevent gradient underflow
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)          # unscale before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)              # skips step on overflow
        scaler.update()                     # adjust scale factor
    else:
        # BF16 path: no scaling needed
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
```

<span class="hh-tag">Listing 5:</span>Mixed precision training with HuggingFace and manual PyTorch AMP.

DeepSeek-V3 [71] demonstrated that 671B-parameter training in FP8 (E4M3 for forward pass, E5M2 for backward) with fine-grained tile-wise scaling achieves less than 0.25% relative loss degradation versus BF16—at a total training cost of approximately $5.6M. Key techniques include stochastic rounding to reduce quantization bias, per-tile scaling factors that adapt to local activation magnitudes, and keeping the first and last layers in higher precision where quantization sensitivity is greatest. NVIDIA’s Nemotron 3 pushes further to NVFP4 (4-bit), demonstrating stability over 25T training tokens by maintaining approximately 15% of layers in BF16. The economic implication is profound: careful precision engineering reduces frontier training costs by 3–5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>, making billion-parameter training accessible beyond the largest labs.

### Practical Optimizer Settings by Training Phase

## Flash Attention – Algorithm and Hardware Awareness

Flash Attention [64, 65] is one of the most impactful algorithmic innovations in deep learning since the transformer itself. It does not change the mathematical result of attention – it computes <em>exactly</em> the same output – but it restructures the memory access pattern so that the GPU’s limited fast SRAM does all the heavy lifting, cutting HBM footprint from <math alttext="O(n^{2})" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>n</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math> to <math alttext="O(n)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>n</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> and delivering 2–4<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> end-to-end wall-clock gains on typical workloads.

### The Standard Attention Memory Problem

Standard scaled dot-product attention is:

<div class="hh-equation" id="ch1.ex29"><math alttext="\text{Attention}(Q,K,V)=\text{softmax}\!\left(\frac{QK^{T}}{\sqrt{d_{k}}}\right)V" display="block"><semantics><mrow><mrow><mtext>Attention</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo>(</mo><mfrac><mrow><mi>Q</mi><mo lspace="0em" rspace="0em">​</mo><msup><mi>K</mi><mi>T</mi></msup></mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mfrac><mo>)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mi>V</mi></mrow></mrow></semantics></math></div>

### The Flash Attention Key Insight – Tiling and Online Softmax

The core insight is: we never need the full <math alttext="n\times n" display="inline"><semantics><mrow><mi>n</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>n</mi></mrow></semantics></math> matrix in memory at once. We can compute the output <math alttext="O" display="inline"><semantics><mi>O</mi></semantics></math> block-by-block if we use the <em>online softmax</em> trick.

Recall that softmax requires a global maximum for numerical stability:

<div class="hh-equation" id="ch1.ex30"><math alttext="\text{softmax}(x_{i})=\frac{e^{x_{i}-m}}{\sum_{j}e^{x_{j}-m}},\quad m=\max_{j}x_{j}" display="block"><semantics><mrow><mrow><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><msup><mi>e</mi><mrow><msub><mi>x</mi><mi>i</mi></msub><mo>−</mo><mi>m</mi></mrow></msup><mrow><msub><mo>∑</mo><mi>j</mi></msub><msup><mi>e</mi><mrow><msub><mi>x</mi><mi>j</mi></msub><mo>−</mo><mi>m</mi></mrow></msup></mrow></mfrac></mrow><mo rspace="1.167em">,</mo><mrow><mi>m</mi><mo>=</mo><mrow><munder><mi>max</mi><mi>j</mi></munder><mo lspace="0.167em">⁡</mo><msub><mi>x</mi><mi>j</mi></msub></mrow></mrow></mrow></semantics></math></div>

The trick: we can <em>update</em> the running maximum and normalization factor as we process new blocks, without ever materializing the full row.

### The Flash Attention Algorithm

### Flash Attention 2 – Better Parallelism

Flash Attention 2 [65] made three key improvements:

<ol><li>Reduced non-matmul FLOPs: The original FA had unnecessary rescaling operations in the inner loop. FA2 restructures the loop to minimize these. On A100, Tensor Core matrix multiplications outpace scalar operations by roughly 16<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math>, so even a small fraction of non-matmul work in the inner loop becomes the latency bottleneck.</li><li>Better parallelism across sequence dimension: FA1 parallelized over batch and heads only. FA2 also parallelizes over the query sequence dimension, enabling better GPU utilization for long sequences with small batch sizes.</li><li>Causal masking optimization: For autoregressive (causal) attention, roughly half the blocks are fully masked. FA2 skips these blocks entirely, giving <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> speedup for causal attention vs. bidirectional.</li></ol>

### Flash Attention 3 – Hopper Architecture

Flash Attention 3 [322] is designed specifically for H100 and exploits three Hopper-specific features:

<ul><li>TMA (Tensor Memory Accelerator): H100 has a dedicated hardware unit for asynchronous bulk data movement between HBM and SRAM. FA3 uses TMA to overlap data loading with computation, hiding memory latency.</li><li>Warp-specialization: FA3 assigns different warps to different roles (producer warps load data via TMA; consumer warps compute MMA). This is a software pipelining technique that keeps both the memory system and Tensor Cores busy simultaneously.</li><li>FP8 support: H100 supports FP8 (E4M3/E5M2) Tensor Core operations at 2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> the throughput of BF16. FA3 supports FP8 attention with per-block quantization to maintain accuracy.</li></ul>

FA3 achieves up to 75% of H100 theoretical peak for FP16 attention, compared to <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>35% for FA2.

### Flash Attention 4 – Blackwell Architecture

Flash Attention 4 [415] targets NVIDIA’s Blackwell GPUs (B200/GB200), which double Tensor Core throughput to 2.25 PFLOP/s (BF16) while non-matmul units (exponential, shared memory bandwidth) scale at a slower rate. This <em>asymmetric hardware scaling</em> means that the bottleneck shifts: on Blackwell, attention is limited not by matmul but by the softmax exponentials and shared memory traffic surrounding them.

FA4 addresses this with four key techniques:

<ul><li>Fully asynchronous MMA pipelines: Blackwell’s MMA instructions are fully asynchronous (unlike Hopper’s wgmma which still blocked on completion). FA4 redesigns the pipeline to overlap MMA, TMA loads, and softmax rescaling across larger tile sizes, keeping all hardware units saturated.</li><li>Software-emulated exponential: Instead of calling the hardware <code>ex2</code> unit (which is the throughput bottleneck), FA4 emulates <math alttext="e^{x}" display="inline"><semantics><msup><mi>e</mi><mi>x</mi></msup></semantics></math> using polynomial approximations executed on the much faster Tensor Cores themselves. This trades extra matmul instructions for exponential-unit stalls.</li><li>Conditional softmax rescaling: Standard FlashAttention rescales the running <math alttext="\max" display="inline"><semantics><mi>max</mi></semantics></math> every tile. FA4 skips the rescaling when the new tile’s max does not exceed the running max (common in practice), saving both register shuffles and synchronization barriers.</li><li>Tensor Memory + 2-CTA MMA mode (backward pass): The backward pass uses Blackwell’s <em>Tensor Memory</em> (a per-SM scratchpad larger than shared memory) and a 2-CTA cooperative mode that fuses <math alttext="dQ" display="inline"><semantics><mrow><mi>d</mi><mo lspace="0em" rspace="0em">​</mo><mi>Q</mi></mrow></semantics></math> accumulation across two thread-block clusters, halving shared memory round-trips.</li></ul>

On B200 with BF16 head-dim 128 (causal, seq-len 8K):

<ul><li>1613 TFLOP/s – 71% of Blackwell peak utilization</li><li>1.3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> faster than cuDNN 9.13 (NVIDIA’s proprietary fused kernel)</li><li>2.7<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> faster than Triton on the same hardware</li></ul>

## Pretraining: Best Practices

Pretraining is the most expensive phase of LLM development—consuming millions of GPU-hours and requiring careful orchestration of data, compute, and hyperparameters. This section distills key lessons from Llama-3 [118], Chinchilla [140], and GPT-4 [274].

### Training Objective

All modern decoder-only LLMs use causal language modeling (CLM):

<div class="hh-equation" id="ch1.ex31"><math alttext="\mathcal{L}_{\text{CLM}}=-\frac{1}{T}\sum_{t=1}^{T}\log P_{\theta}(x_{t}\mid x_{&lt;t})" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>CLM</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>T</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mi>log</mi><msub><mi>P</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo lspace="0em" rspace="0.167em">∣</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math></div>

This simple objective—with enough data and scale—produces emergent capabilities (in-context learning, reasoning, instruction following) without explicit supervision [32].

### Data Pipeline

### Scaling Laws

Hoffmann et al. [140] showed that compute-optimal training requires balancing model size <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> and data size <math alttext="D" display="inline"><semantics><mi>D</mi></semantics></math>: <math alttext="N_{\text{opt}}\propto C^{0.50}" display="inline"><semantics><mrow><msub><mi>N</mi><mtext>opt</mtext></msub><mo>∝</mo><msup><mi>C</mi><mn>0.50</mn></msup></mrow></semantics></math>, <math alttext="D_{\text{opt}}\propto C^{0.50}" display="inline"><semantics><mrow><msub><mi>D</mi><mtext>opt</mtext></msub><mo>∝</mo><msup><mi>C</mi><mn>0.50</mn></msup></mrow></semantics></math>. A 70B model is compute-optimal at <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1.4T tokens. In practice, models are <em>over-trained</em> (more tokens than Chinchilla-optimal) because inference cost scales with model size, not training tokens—smaller over-trained models are cheaper to deploy.

### Key Hyperparameters

<div class="hh-table" id="ch1.t10"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Setting</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Llama-3 405B</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Llama-3 8B</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Qwen-2.5 72B</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Mistral 7B</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Tokens</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>15T</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>15T</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>18T</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8T</span></span></td></tr><tr><th class="ltx_align_left">Batch size (tokens)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>16M</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4M</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4M</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4M</span></span></td></tr><tr><th class="ltx_align_left">Peak LR</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="8\text{e-}5" display="inline"><semantics><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>5</mn></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="3\text{e-}4" display="inline"><semantics><mrow><mn>3</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>4</mn></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="3\text{e-}4" display="inline"><semantics><mrow><mn>3</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>4</mn></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="3\text{e-}4" display="inline"><semantics><mrow><mn>3</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>4</mn></mrow><span></span></semantics></math></span></span></td></tr><tr><th class="ltx_align_left">Schedule</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>WSD</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>WSD</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cosine</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cosine</span></span></td></tr><tr><th class="ltx_align_left">Weight decay</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>0.1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>0.1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>0.1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>0.1</span></span></td></tr><tr><th class="ltx_align_left">Context length</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8192</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8192</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4096<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo><span></span></semantics></math>32K</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8192</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.10:</span>Pretraining hyperparameters from published models.</p></div>

### Common Failure Modes

### Mid-Training: Preparing for Reinforcement Learning

A distinct pipeline stage has emerged between raw pretraining and post-training. <em>Mid-training</em> (sometimes called “continued pretraining” or “annealing”) up-weights STEM, mathematics, and code on high-quality curated data while extending context length to 128K–256K tokens. Its purpose is not to teach new knowledge but to <em>prepare the model for RL</em>: instilling the cognitive behaviors—verification, backtracking, subgoal decomposition—that reinforcement learning will later amplify.

Typical mid-training uses 100B–500B tokens of curated high-quality data with a cosine-decayed learning rate. Llama 3’s “annealing tail,” MiMo’s 3-stage mixture (ramping math+code to 70%), and MAI-Thinking-1’s explicit STEM up-weight are all instances of this pattern.

## Supervised Fine-Tuning (SFT)

SFT transforms a pretrained language model into an instruction-following assistant by training on curated prompt–response pairs. This is the bridge between raw language modeling and RLHF.

### SFT Objective

The loss is identical to CLM, but computed only on response tokens:

<div class="hh-equation" id="ch1.ex32"><math alttext="\mathcal{L}_{\text{SFT}}=-\frac{1}{|y|}\sum_{t=1}^{|y|}\log P_{\theta}(y_{t}\mid x_{\text{prompt}},y_{&lt;t})" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>SFT</mtext></msub><mo rspace="0em">=</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></munderover><mi>log</mi><msub><mi>P</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0.167em">∣</mo><msub><mi>x</mi><mtext>prompt</mtext></msub><mo>,</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math></div>

Prompt tokens provide context but receive no gradient (labels set to <math alttext="-100" display="inline"><semantics><mrow><mo>−</mo><mn>100</mn></mrow></semantics></math>).

### Data Quality: The LIMA Principle

Zhou et al. [439] demonstrated that 1,000 carefully curated examples can match models trained on 50K+ noisy examples. Key requirements:

<ul><li>Diversity: Cover QA, summarization, code, math, creative writing, multi-turn dialogue</li><li>Correctness: Every response must be factually accurate and well-formatted</li><li>Length balance: Mix short (1-sentence) and long (multi-paragraph) responses</li><li>Decontamination: Remove overlap with evaluation benchmarks</li></ul>

### Training Configuration

```python
from trl import SFTTrainer, SFTConfig

sft_config = SFTConfig(
    output_dir="./sft_output",
    max_seq_length=4096,
    packing=True,              # Pack short examples into full sequences
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    gradient_checkpointing=True,
)
trainer = SFTTrainer(model=model, args=sft_config,
                     train_dataset=dataset, processing_class=tokenizer)
trainer.train()
```

### Efficient Training Solutions

Standard HuggingFace training leaves significant performance on the table. Several libraries provide drop-in efficiency gains for SFT workloads:

An open-source set of Triton-fused kernels from LinkedIn that replace standard PyTorch operators during training. Key fusions include:

<ul><li>Fused Cross-Entropy: Merges the final linear projection, softmax, and loss computation into a single kernel—avoids materializing the full <math alttext="(\text{batch}\times\text{seq}\times\text{vocab})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mrow><mtext>batch</mtext><mo lspace="0.222em" rspace="0.222em">×</mo><mtext>seq</mtext><mo lspace="0.222em" rspace="0.222em">×</mo><mtext>vocab</mtext></mrow><mo stretchy="false">)</mo></mrow></semantics></math> logit tensor.</li><li>Fused RMSNorm / SwiGLU / RoPE: Eliminates intermediate memory allocations for common LLM building blocks.</li><li>Chunked operations: Processes large tensors in tiles to keep peak memory bounded.</li></ul>

Result: 20% higher throughput and up to 60% memory reduction with a one-line integration (<code>apply_liger_kernel_to_llama()</code>). Compatible with FSDP, DeepSpeed, and LoRA.

A specialized fine-tuning library that combines custom CUDA/Triton kernels with aggressive memory optimization:

<ul><li>Manual backpropagation through LoRA layers (avoids autograd overhead).</li><li>4-bit QLoRA with fused dequantization—trains 70B models on a single 48 GB GPU.</li><li>Intelligent RoPE and attention kernel fusion specific to each architecture (Llama, Mistral, Qwen, Gemma).</li></ul>

Result: 2–5<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> faster than vanilla HuggingFace + PEFT, with 60–70% less VRAM. Particularly impactful for single-GPU and consumer-hardware workflows.

Meta’s native PyTorch fine-tuning library (development wound down in 2025), designed around composability rather than monolithic abstractions:

<ul><li>Pure PyTorch—no trainer class; recipes are readable single-file scripts.</li><li>Native integration with <code>torch.compile</code>, FSDP2, and activation checkpointing.</li><li>First-class support for QLoRA, full fine-tuning, and knowledge distillation.</li><li>Built-in quantization-aware training (QAT) for post-training compression.</li></ul>

Result: Comparable speed to custom solutions but with full debuggability and no framework lock-in.

### Best Practices

<div class="hh-table" id="ch1.t11"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Practice</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Details</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Packing</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Concatenate multiple short examples into one sequence (separated by EOS). Avoids padding waste.</span></span></td></tr><tr><th class="ltx_align_left">NEFTune <span>[159]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Add uniform noise to embeddings (<math alttext="\alpha=5" display="inline"><semantics><mrow><mi>α</mi><mo>=</mo><mn>5</mn></mrow><span></span></semantics></math>). Improves MT-Bench by 5–15% at zero cost.</span></span></td></tr><tr><th class="ltx_align_left">Chat template</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Always use the model’s native template. Mismatched templates degrade quality.</span></span></td></tr><tr><th class="ltx_align_left">Epochs</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2–3 for large datasets; up to 5 for small (<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo><span></span></semantics></math>10K) curated sets. Over-training causes format memorization.</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.11:</span>SFT training guidelines.</p></div>

## LoRA and Parameter-Efficient Fine-Tuning

Full fine-tuning of a 70B model requires storing 70B trainable parameters plus their optimizer states (560+ GB of memory). LoRA [146] (Low-Rank Adaptation) provides a way to fine-tune with <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>1% of the parameters while achieving comparable quality.

### The LoRA Insight

<figure id="ch1.f10"><img src="./fig_011_lora-decomposition.png" alt="Figure 1.10: LoRA decomposes the weight update Δ​W\Delta W into two small matrices B×AB\times A. The original weight WW " loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.10:</span>LoRA decomposes the weight update <math alttext="\Delta W" display="inline"><semantics><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>W</mi></mrow></semantics></math> into two small matrices <math alttext="B\times A" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>A</mi></mrow></semantics></math>. The original weight <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> remains frozen; only <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> and <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> receive gradients. At inference, the product <math alttext="BA" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></semantics></math> can be merged into <math alttext="W" display="inline"><semantics><mi>W</mi></semantics></math> with zero overhead.</figcaption></figure>

### LoRA Hyperparameters

Choosing LoRA hyperparameters correctly is critical — the wrong rank or alpha can either under-fit (too constrained) or waste memory (too expressive).

<div class="hh-table" id="ch1.t12"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Hyperparameter</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Typical Values</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Guidance</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>r</span> (rank)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>8, 16, 32, 64</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Higher = more capacity but more memory. Start with 16.</span></span></td></tr><tr><th class="ltx_align_left"><span>lora_alpha</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>16, 32 (often <math alttext="=r" display="inline"><semantics><mrow><mphantom></mphantom><mo>=</mo><mi>r</mi></mrow><span></span></semantics></math> or <math alttext="2r" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>r</mi></mrow><span></span></semantics></math>)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Controls update magnitude via <math alttext="\alpha/r" display="inline"><semantics><mrow><mi>α</mi><mo>/</mo><mi>r</mi></mrow><span></span></semantics></math> scaling.</span></span></td></tr><tr><th class="ltx_align_left"><span>target_modules</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>q_proj, k_proj, v_proj, o_proj</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>All attention projections. Add <span>gate_proj, up_proj, down_proj</span> for full coverage.</span></span></td></tr><tr><th class="ltx_align_left"><span>lora_dropout</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>0.0–0.1</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Regularization. Usually 0.05 for small datasets.</span></span></td></tr><tr><th class="ltx_align_left"><span>bias</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>"none"</span></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Training biases adds minimal params but rarely helps.</span></span></td></tr><tr><th class="ltx_align_left">Learning rate</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="1\text{e-}4" display="inline"><semantics><mrow><mn>1</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>4</mn></mrow><span></span></semantics></math> to <math alttext="3\text{e-}4" display="inline"><semantics><mrow><mn>3</mn><mo lspace="0em" rspace="0em">​</mo><mtext>e-</mtext><mo lspace="0em" rspace="0em">​</mo><mn>4</mn></mrow><span></span></semantics></math></span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Higher than full fine-tuning (only adapters update).</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.12:</span>LoRA hyperparameter guide.</p></div>

### LoRA Variants

<div class="hh-table" id="ch1.t13"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Key Innovation</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>When to Use</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>QLoRA</span><span>[75]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>4-bit quantized base + LoRA in BF16. NF4 data type + double quantization.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Fine-tune 70B on single 48GB GPU.</span></span></td></tr><tr><th class="ltx_align_left"><span>DoRA</span><span>[229]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Decomposes <math alttext="W" display="inline"><semantics><mi>W</mi><span></span></semantics></math> into magnitude and direction; LoRA updates direction only.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Better generalization for reasoning.</span></span></td></tr><tr><th class="ltx_align_left"><span>LoRA+</span><span>[136]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Different LRs for <math alttext="A" display="inline"><semantics><mi>A</mi><span></span></semantics></math>/<math alttext="B" display="inline"><semantics><mi>B</mi><span></span></semantics></math> (<math alttext="\eta_{B}=\lambda\eta_{A}" display="inline"><semantics><mrow><msub><mi>η</mi><mi>B</mi></msub><mo>=</mo><mrow><mi>λ</mi><mo lspace="0em" rspace="0em">​</mo><msub><mi>η</mi><mi>A</mi></msub></mrow></mrow><span></span></semantics></math>, <math alttext="\lambda\approx 16" display="inline"><semantics><mrow><mi>λ</mi><mo>≈</mo><mn>16</mn></mrow><span></span></semantics></math>).</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Free 2% gain; no extra cost.</span></span></td></tr><tr><th class="ltx_align_left"><span>AdaLoRA</span><span>[424]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Dynamic rank budget across layers (SVD-based importance).</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Very tight compute budget.</span></span></td></tr><tr><th class="ltx_align_left"><span>rsLoRA</span><span>[176]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Scales by <math alttext="\alpha/\sqrt{r}" display="inline"><semantics><mrow><mi>α</mi><mo>/</mo><msqrt><mi>r</mi></msqrt></mrow><span></span></semantics></math> instead of <math alttext="\alpha/r" display="inline"><semantics><mrow><mi>α</mi><mo>/</mo><mi>r</mi></mrow><span></span></semantics></math>. Stable at high ranks.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>When using <math alttext="r\geq 64" display="inline"><semantics><mrow><mi>r</mi><mo>≥</mo><mn>64</mn></mrow><span></span></semantics></math>.</span></span></td></tr><tr><th class="ltx_align_left"><span>VeRA</span><span>[190]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Shared frozen random <math alttext="A,B" display="inline"><semantics><mrow><mi>A</mi><mo>,</mo><mi>B</mi></mrow><span></span></semantics></math>; trains diagonal scaling only.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Extreme param efficiency.</span></span></td></tr><tr><th class="ltx_align_left"><span>LoRA-FA</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Freezes <math alttext="A" display="inline"><semantics><mi>A</mi><span></span></semantics></math> after init; only trains <math alttext="B" display="inline"><semantics><mi>B</mi><span></span></semantics></math>. Halves LoRA memory.</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Memory-constrained scenarios.</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.13:</span>LoRA variants and their innovations.</p></div>

DoRA [229] observes that full fine-tuning tends to change the <em>direction</em> of weight vectors more than their magnitude. Standard LoRA conflates both. DoRA decomposes each weight column into magnitude <math alttext="m=\|W\|_{\text{col}}" display="inline"><semantics><mrow><mi>m</mi><mo>=</mo><msub><mrow><mo stretchy="false">‖</mo><mi>W</mi><mo stretchy="false">‖</mo></mrow><mtext>col</mtext></msub></mrow></semantics></math> and direction <math alttext="\hat{V}=W/\|W\|_{\text{col}}" display="inline"><semantics><mrow><mover accent="true"><mi>V</mi><mo>^</mo></mover><mo>=</mo><mrow><mi>W</mi><mo>/</mo><msub><mrow><mo stretchy="false">‖</mo><mi>W</mi><mo stretchy="false">‖</mo></mrow><mtext>col</mtext></msub></mrow></mrow></semantics></math>, then applies LoRA only to the direction:

<div class="hh-equation" id="ch1.ex35"><math alttext="W^{\prime}=m\odot\hat{V}^{\prime},\quad\hat{V}^{\prime}=\frac{W+BA}{\|W+BA\|_{\text{col}}}" display="block"><semantics><mrow><mrow><msup><mi>W</mi><mo>′</mo></msup><mo>=</mo><mrow><mi>m</mi><mo lspace="0.222em" rspace="0.222em">⊙</mo><msup><mover accent="true"><mi>V</mi><mo>^</mo></mover><mo>′</mo></msup></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msup><mover accent="true"><mi>V</mi><mo>^</mo></mover><mo>′</mo></msup><mo>=</mo><mfrac><mrow><mi>W</mi><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><msub><mrow><mo stretchy="false">‖</mo><mrow><mi>W</mi><mo>+</mo><mrow><mi>B</mi><mo lspace="0em" rspace="0em">​</mo><mi>A</mi></mrow></mrow><mo stretchy="false">‖</mo></mrow><mtext>col</mtext></msub></mfrac></mrow></mrow></semantics></math></div>

Magnitude <math alttext="m" display="inline"><semantics><mi>m</mi></semantics></math> is a separate learnable vector (one scalar per column). This consistently outperforms LoRA by 1–3% on reasoning and instruction-following benchmarks with no additional inference cost (merged at deployment).

Hayou et al. [136] show that matrices <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> and <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> in LoRA have different optimal learning rates. Since <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> is initialized to zero, it starts in a very different regime than <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> (initialized from <math alttext="\mathcal{N}(0,\sigma^{2})" display="inline"><semantics><mrow><mi>𝒩</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><msup><mi>σ</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></semantics></math>). Setting <math alttext="\eta_{B}\approx 16\times\eta_{A}" display="inline"><semantics><mrow><msub><mi>η</mi><mi>B</mi></msub><mo>≈</mo><mrow><mn>16</mn><mo lspace="0.222em" rspace="0.222em">×</mo><msub><mi>η</mi><mi>A</mi></msub></mrow></mrow></semantics></math> improves convergence speed and final quality by <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>2% — a free gain requiring only a one-line config change:

```python
# LoRA+ in PEFT: set different LRs per matrix
optimizer_grouped_parameters = [
    {"params": [p for n, p in model.named_parameters() if "lora_B" in n],
     "lr": 2e-4 * 16},   # B matrix: higher LR
    {"params": [p for n, p in model.named_parameters() if "lora_A" in n],
     "lr": 2e-4},         # A matrix: base LR
]
```

VeRA [190] takes parameter efficiency to the extreme: instead of learning <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> and <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math>, it <em>freezes</em> them as shared random matrices across all layers and only trains two diagonal scaling vectors <math alttext="d_{b}\in\mathbb{R}^{r}" display="inline"><semantics><mrow><msub><mi>d</mi><mi>b</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mi>r</mi></msup></mrow></semantics></math> and <math alttext="d_{a}\in\mathbb{R}^{d}" display="inline"><semantics><mrow><msub><mi>d</mi><mi>a</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mi>d</mi></msup></mrow></semantics></math>:

<div class="hh-equation" id="ch1.ex36"><math alttext="\Delta W=B\cdot\text{diag}(d_{b})\cdot A\cdot\text{diag}(d_{a})" display="block"><semantics><mrow><mrow><mi mathvariant="normal">Δ</mi><mo lspace="0em" rspace="0em">​</mo><mi>W</mi></mrow><mo>=</mo><mrow><mrow><mrow><mi>B</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>diag</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>d</mi><mi>b</mi></msub><mo rspace="0.055em" stretchy="false">)</mo></mrow></mrow><mo rspace="0.222em">⋅</mo><mi>A</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mtext>diag</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>d</mi><mi>a</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math></div>

This reduces trainable parameters by <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>10<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> vs. LoRA (only <math alttext="r+d" display="inline"><semantics><mrow><mi>r</mi><mo>+</mo><mi>d</mi></mrow></semantics></math> params per layer) while achieving 90–95% of LoRA quality. Best for scenarios where you need hundreds of task-specific adapters with minimal storage.

```python
# QLoRA configuration with PEFT
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 - optimal for weights
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,       # Quantize the quantization constants
)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,                        # alpha/r = 2x scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)  # Prepare for QLoRA
model = get_peft_model(model, lora_config)       # Add LoRA adapters
model.print_trainable_parameters()
# Output: trainable params: 83,886,080 || all params: 70,553,706,496 || 0.12%
```

### Other PEFT Approaches

LoRA dominates modern practice, but it is not the only parameter-efficient method. For completeness, the main alternatives:

<div class="hh-table" id="ch1.t14"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Mechanism</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Pros / Cons</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Status</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>LoRA</span><span>[146]</span> (and variants)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Low-rank matrices added to existing weights</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Mergeable at inference (zero overhead); well-supported; works for all architectures</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Standard</span></span></span></td></tr><tr><th class="ltx_align_left"><span>Adapters</span><span>[144]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Small bottleneck MLPs inserted between layers</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Modular; stackable; adds inference latency (extra sequential layers)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Rarely used</span></span></td></tr><tr><th class="ltx_align_left"><span>Prefix Tuning</span><span>[212]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learnable “virtual tokens” prepended to keys/values at each layer</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No weight modification; effective for generation tasks; consumes context length</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Niche</span></span></td></tr><tr><th class="ltx_align_left"><span>Prompt Tuning</span><span>[205]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learnable soft prompt embeddings prepended to input</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Extremely few params (<math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo><span></span></semantics></math>0.01%); weaker than LoRA for complex tasks</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Niche</span></span></td></tr><tr><th class="ltx_align_left"><span>IA3</span><span>[223]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Learned vectors that rescale keys, values, and FFN activations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Even fewer params than LoRA; mergeable; limited capacity</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Deprecated</span></span></td></tr><tr><th class="ltx_align_left"><span>BitFit</span><span>[417]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Train only bias terms</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Near-zero params; surprisingly effective for simple tasks; limited expressiveness</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Historical</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.14:</span>PEFT method families. LoRA is the de facto standard for LLM fine-tuning; the others are included for historical context and niche use cases.</p></div>

## Mixture of Experts (MoE)

Mixture of Experts models [325, 166] scale model capacity without proportionally scaling compute cost by activating only a subset of parameters for each token.

### Architecture

<figure id="ch1.f11"><img src="./fig_012_fig12.png" alt="Figure 1.11: MoE layer with 8 experts and Top-2 routing. Only the two highest-gated experts are computed per token; the " loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.11:</span>MoE layer with 8 experts and Top-2 routing. Only the two highest-gated experts are computed per token; the rest are skipped entirely.</figcaption></figure>

### Load Balancing

### Noisy Top-K Gating: Making Discrete Routing Trainable

The core challenge in MoE is that top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> selection is not differentiable — you can’t backpropagate through a hard “pick the top 2” operation. The field has developed two key tricks to solve this:

Add learnable Gaussian noise to the router logits <em>before</em> the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> selection:



<ul><li><math alttext="W_{\text{noise}}" display="inline"><semantics><msub><mi>W</mi><mtext>noise</mtext></msub></semantics></math> is a <em>learned</em> noise magnitude — the model learns how much exploration each expert needs</li><li>During training, noise occasionally promotes “underdog” experts into the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>, giving them gradient signal</li><li>At inference, noise is removed: use clean logits <math alttext="h(x)" display="inline"><semantics><mrow><mi>h</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math> for deterministic routing</li><li>The Softplus ensures noise scale is always positive</li></ul>

An alternative from the variational inference literature [161]. The Gumbel-Max trick provides exact sampling from a categorical distribution:

<div class="hh-equation" id="ch1.e12"><math alttext="z=\arg\max_{i}\left[\log\pi_{i}+G_{i}\right],\quad G_{i}\sim\text{Gumbel}(0,1)" display="block"><semantics><mrow><mrow><mi>z</mi><mo>=</mo><mrow><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><munder><mi>max</mi><mi>i</mi></munder></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>i</mi></msub></mrow><mo>+</mo><msub><mi>G</mi><mi>i</mi></msub></mrow><mo>]</mo></mrow></mrow></mrow><mo rspace="1.167em">,</mo><mrow><msub><mi>G</mi><mi>i</mi></msub><mo>∼</mo><mrow><mtext>Gumbel</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(1.12)</span></div>

where Gumbel noise is generated as <math alttext="G_{i}=-\log(-\log(U_{i})),\;U_{i}\sim\text{Uniform}(0,1)" display="inline"><semantics><mrow><mrow><msub><mi>G</mi><mi>i</mi></msub><mo>=</mo><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mo rspace="0.167em">−</mo><mrow><mi>log</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>U</mi><mi>i</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo rspace="0.447em">,</mo><mrow><msub><mi>U</mi><mi>i</mi></msub><mo>∼</mo><mrow><mtext>Uniform</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>.

For top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> routing: taking the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> of <math alttext="(\log\pi_{i}+G_{i})" display="inline"><semantics><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>i</mi></msub></mrow><mo>+</mo><msub><mi>G</mi><mi>i</mi></msub></mrow><mo stretchy="false">)</mo></mrow></semantics></math> gives <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> samples <em>without replacement</em> from the categorical distribution defined by <math alttext="\pi" display="inline"><semantics><mi>π</mi></semantics></math>.

Since <math alttext="\arg\max" display="inline"><semantics><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mi>max</mi></mrow></semantics></math> is non-differentiable, the Gumbel-Softmax relaxation replaces it with a temperature-controlled softmax:

<div class="hh-equation" id="ch1.e13"><math alttext="\hat{g}_{i}=\frac{\exp\left((\log\pi_{i}+G_{i})/\tau\right)}{\sum_{j}\exp\left((\log\pi_{j}+G_{j})/\tau\right)}" display="block"><semantics><mrow><msub><mover accent="true"><mi>g</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>i</mi></msub></mrow><mo>+</mo><msub><mi>G</mi><mi>i</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mi>τ</mi></mrow><mo>)</mo></mrow></mrow><mrow><msub><mo>∑</mo><mi>j</mi></msub><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo>(</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>π</mi><mi>j</mi></msub></mrow><mo>+</mo><msub><mi>G</mi><mi>j</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mi>τ</mi></mrow><mo>)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(1.13)</span></div>

<ul><li><math alttext="\tau\to 0" display="inline"><semantics><mrow><mi>τ</mi><mo stretchy="false">→</mo><mn>0</mn></mrow></semantics></math>: approaches a hard one-hot (exact but non-differentiable)</li><li><math alttext="\tau\to\infty" display="inline"><semantics><mrow><mi>τ</mi><mo stretchy="false">→</mo><mi mathvariant="normal">∞</mi></mrow></semantics></math>: approaches uniform (differentiable but uninformative)</li><li>In practice, anneal <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> from 1.0 down to 0.1–0.5 during training</li><li>Straight-through estimator: use hard top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> in the forward pass, but Gumbel-Softmax gradients in the backward pass — best of both worlds</li></ul>

The standard approach adds a balance penalty to the training loss, but DeepSeek [372] demonstrated that this fights the routing signal: the loss pushes toward uniform utilization while the router tries to specialize. Their solution moves balancing entirely out of the gradient into a per-expert routing <em>bias</em> updated based on observed utilization—achieving both better quality <em>and</em> more expert specialization. A subsequent insight from Qwen revealed that the <em>aggregation scope</em> matters more than the mechanism: computing balance statistics per micro-batch (the common default) silently destroys specialization because micro-batches are too homogeneous. Aggregating over the full global batch—as MAI-Thinking-1 confirms—restores the diversity signal that experts need to differentiate.

### Notable MoE Models

ModelTotal ParamsActive ParamsExpertsInnovationSwitch Transformer [89]1.6T100B128, Top-1First large-scale MoE; simplified routingMixtral 8x7B [166]47B13B8, Top-2Open-weight; matches Llama-2 70B qualityDeepSeek-V2 [70]236B21B160, Top-6DeepSeekMoE with shared + routed expertsQwen-MoE [350]14.3B2.7B60, Top-4Fine-grained experts for efficiencyDBRX [67]132B36B16, Top-4Fine-grained with 4 experts per block

## Diversity in LLM Training

Diversity — in training data, model outputs, and optimization trajectories — is critical for preventing mode collapse and ensuring robust, general-purpose LLMs. This section covers the key diversity mechanisms applicable to all LLM training phases.

### Sampling Diversity

### Training Data Diversity

<ul><li>Prompt diversity: Cover different domains, difficulty levels, and formats. The Goldilocks principle: prompts should have 20–80% success rate.</li><li>Deduplication: Remove near-duplicate training examples (MinHash, n-gram overlap). Duplicates cause overfitting to specific patterns.</li><li>Data mixing: Balance across tasks/domains using temperature-weighted sampling or curriculum strategies.</li></ul>

### Diversity-Promoting Methods

MethodHow It Promotes DiversityTemperature scalingHigher <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> flattens the distribution; more tokens become plausible.Top-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> / Min-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math>Adaptive thresholds allow wider sampling when the model is uncertain.Frequency penaltyPenalizes repeated tokens, forcing lexical variety within a response.Data deduplicationRemoving near-duplicates from training data prevents overfitting to specific patterns.Multi-domain mixingTemperature-weighted sampling across domains ensures broad coverage.Verbalized samplingPrompt the model to explicitly verbalize a probability distribution over responses [422]. See §7.5 GRPO Variants and Extensions.

## Text Generation: Decoding Methods

A trained language model outputs a probability distribution over the vocabulary at each step: <math alttext="P(x_{t}|x_{&lt;t})" display="inline"><semantics><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math>. The decoding strategy determines how we select the next token from this distribution. This choice profoundly affects output quality, diversity, and coherence.

### Greedy Decoding

The simplest strategy: always pick the highest-probability token.

<div class="hh-equation" id="ch1.ex42"><math alttext="x_{t}=\arg\max_{v\in\mathcal{V}}P(v|x_{&lt;t})" display="block"><semantics><mrow><msub><mi>x</mi><mi>t</mi></msub><mo>=</mo><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mrow><munder><mi>max</mi><mrow><mi>v</mi><mo>∈</mo><mi>𝒱</mi></mrow></munder><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math></div>

Intuition: Like always taking the most obvious next word in a sentence. “The capital of France is…” <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> “Paris” (probability 0.92).

Pros: Deterministic, fast, no hyperparameters. <br>

Cons: Produces repetitive, generic text. Misses high-quality sequences where an early low-probability token leads to a globally better output. No diversity.

### Beam Search

Maintain <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> (beam width) partial hypotheses in parallel, expanding each by the top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> tokens and keeping the <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> highest-scoring complete sequences:

<div class="hh-equation" id="ch1.ex43"><math alttext="\text{score}(y_{1:t})=\sum_{i=1}^{t}\log P(y_{i}|y_{&lt;i})" display="block"><semantics><mrow><mtext>score</mtext><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mrow><mn>1</mn><mo lspace="0.278em" rspace="0.278em">:</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow><mo rspace="0.111em">=</mo><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>t</mi></munderover><mi>log</mi><mi>P</mi><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo fence="false" rspace="0.167em" stretchy="false">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math></div>

With length normalization to avoid favoring short sequences:

<div class="hh-equation" id="ch1.ex44"><math alttext="\text{score}_{\text{norm}}(y)=\frac{1}{|y|^{\alpha}}\sum_{i=1}^{|y|}\log P(y_{i}|y_{&lt;i}),\quad\alpha\in[0.6,1.0]" display="block"><semantics><mrow><mrow><mrow><msub><mtext>score</mtext><mtext>norm</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>y</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mfrac><mn>1</mn><msup><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow><mi>α</mi></msup></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mrow><mo stretchy="false">|</mo><mi>y</mi><mo stretchy="false">|</mo></mrow></munderover><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>i</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>i</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></mrow><mo rspace="1.167em">,</mo><mrow><mi>α</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.6</mn><mo>,</mo><mn>1.0</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></semantics></math></div>

Intuition: Like exploring multiple paths in a maze simultaneously, keeping only the <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> most promising ones at each junction.

Pros: Finds higher-likelihood sequences than greedy; good for translation and summarization where there’s a single “correct” output. <br>

Cons: Still tends toward generic, repetitive text for open-ended generation; <math alttext="B\times" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> more compute; all beams often converge to similar outputs.

<figure id="ch1.f12"><img src="./fig_013_fig13.png" alt="Figure 1.12: Beam search with B=2B=2. At each step, only the 2 highest-scoring partial sequences survive (blue). Lower-s" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.12:</span>Beam search with <math alttext="B=2" display="inline"><semantics><mrow><mi>B</mi><mo>=</mo><mn>2</mn></mrow></semantics></math>. At each step, only the 2 highest-scoring partial sequences survive (blue). Lower-scoring alternatives are pruned (gray).</figcaption></figure>

### Diverse Beam Search

Standard beam search produces near-duplicate beams. Diverse beam search [360] partitions beams into <math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math> groups and adds a dissimilarity penalty between groups:

<div class="hh-equation" id="ch1.ex45"><math alttext="\text{score}_{g}(y_{t})=\log P(y_{t}|y_{&lt;t})-\lambda\sum_{g^{\prime}&lt;g}\Delta(y_{t},Y_{g^{\prime}})" display="block"><semantics><mrow><mrow><msub><mtext>score</mtext><mi>g</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>−</mo><mrow><mi>λ</mi><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><msup><mi>g</mi><mo>′</mo></msup><mo>&lt;</mo><mi>g</mi></mrow></munder><mrow><mi mathvariant="normal">Δ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>y</mi><mi>t</mi></msub><mo>,</mo><msub><mi>Y</mi><msup><mi>g</mi><mo>′</mo></msup></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></mrow></semantics></math></div>

where <math alttext="\Delta" display="inline"><semantics><mi mathvariant="normal">Δ</mi></semantics></math> measures overlap (e.g., Hamming diversity) with tokens already selected by earlier groups, and <math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> controls diversity strength.

Intuition: Like forcing a brainstorming group to generate different ideas — each subgroup is penalized for repeating what earlier subgroups said.

Pros: Produces genuinely different candidate sequences; useful for reranking pipelines. <br>

Cons: Diversity penalty can degrade individual beam quality; more hyperparameters (<math alttext="G" display="inline"><semantics><mi>G</mi></semantics></math>, <math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math>).

### Top- kk Sampling

Sample from only the <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> most probable tokens, redistributing probability mass:

<div class="hh-equation" id="ch1.ex46"><math alttext="P^{\prime}(v|x_{&lt;t})=\begin{cases}\dfrac{P(v|x_{&lt;t})}{\sum_{v^{\prime}\in\text{Top-}k}P(v^{\prime}|x_{&lt;t})}&amp;\text{if }v\in\text{Top-}k\\[6.0pt]
0&amp;\text{otherwise}\end{cases}" display="block"><semantics><mrow><mrow><msup><mi>P</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mfrac><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mo>∑</mo><mrow><msup><mi>v</mi><mo>′</mo></msup><mo>∈</mo><mrow><mtext>Top-</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi></mrow></mrow></msub><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>v</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>v</mi></mrow><mo>∈</mo><mrow><mtext>Top-</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi></mrow></mrow></mtd></mtr><mtr><mtd columnalign="left"><mn>0</mn></mtd><mtd columnalign="left"><mtext>otherwise</mtext></mtd></mtr></mtable></mrow></mrow></semantics></math></div>

Intuition: After “The cat sat on the…”, only consider the top <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> plausible continuations (“mat”, “floor”, “couch”, …) and ignore extremely unlikely ones (“quantum”, “archipelago”).

Pros: Removes tail noise; simple to implement. <br>

Cons: Fixed <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> is too restrictive for peaked distributions (wastes probability mass) and too permissive for flat distributions (lets in garbage tokens).

### Top- pp (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds <math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math>:

<div class="hh-equation" id="ch1.ex47"><math alttext="\text{Top-}p=\min\left\{S\subseteq\mathcal{V}:\sum_{v\in S}P(v|x_{&lt;t})\geq p\right\}" display="block"><semantics><mrow><mrow><mtext>Top-</mtext><mo lspace="0em" rspace="0em">​</mo><mi>p</mi></mrow><mo>=</mo><mrow><mi>min</mi><mo>⁡</mo><mrow><mo>{</mo><mrow><mi>S</mi><mo>⊆</mo><mi>𝒱</mi></mrow><mo lspace="0.278em" rspace="0.111em">:</mo><mrow><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>v</mi><mo>∈</mo><mi>S</mi></mrow></munder><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>≥</mo><mi>p</mi></mrow><mo>}</mo></mrow></mrow></mrow></semantics></math></div>

where tokens are sorted by descending probability and added until the threshold <math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> is reached.

Intuition: Adaptively resize the candidate pool. If the model is confident (“Paris” at 95%), the nucleus is tiny. If uncertain (“The movie was…”), the nucleus expands to include many plausible adjectives.

Pros: Adapts to distribution shape; widely used default (<math alttext="p=0.9" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>0.9</mn></mrow></semantics></math>–<math alttext="0.95" display="inline"><semantics><mn>0.95</mn></semantics></math>). <br>

Cons: Still includes some low-quality tokens at the tail of the nucleus; the threshold is a single global hyperparameter.

<figure id="ch1.f13"><img src="./fig_014_fig14.png" alt="Figure 1.13: Top-pp (nucleus) sampling: tokens are sorted by probability and included until cumulative mass reaches p=0." loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.13:</span>Top-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> (nucleus) sampling: tokens are sorted by probability and included until cumulative mass reaches <math alttext="p=0.9" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>0.9</mn></mrow></semantics></math>. The nucleus (dark blue) adapts its size to the distribution shape — here 5 tokens suffice.</figcaption></figure>

### Min- pp Sampling

A recent alternative that sets a relative probability floor [266]:

<div class="hh-equation" id="ch1.ex48"><math alttext="\text{Min-}p=\left\{v\in\mathcal{V}:P(v|x_{&lt;t})\geq p_{\min}\cdot\max_{v^{\prime}}P(v^{\prime}|x_{&lt;t})\right\}" display="block"><semantics><mrow><mrow><mtext>Min-</mtext><mo lspace="0em" rspace="0em">​</mo><mi>p</mi></mrow><mo>=</mo><mrow><mo>{</mo><mrow><mi>v</mi><mo>∈</mo><mi>𝒱</mi></mrow><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>≥</mo><mrow><msub><mi>p</mi><mi>min</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><munder><mi>max</mi><msup><mi>v</mi><mo>′</mo></msup></munder><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msup><mi>v</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><mo>}</mo></mrow></mrow></semantics></math></div>

Only tokens with probability at least <math alttext="p_{\min}" display="inline"><semantics><msub><mi>p</mi><mi>min</mi></msub></semantics></math> times the top token’s probability are kept.

Intuition: “Only consider tokens that are at least 10% as likely as the best token.” If the top token has probability 0.8, only tokens above 0.08 survive. If the top token has probability 0.05 (very uncertain), tokens above 0.005 survive — naturally expanding the pool.

Pros: Scales naturally with model confidence; fewer degenerate samples than top-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> on peaked distributions; single intuitive parameter. <br>

Cons: Newer, less battle-tested; not yet standard in all inference frameworks.

### Temperature Scaling

Before applying any sampling strategy, logits are divided by temperature <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math>:

<div class="hh-equation" id="ch1.ex49"><math alttext="P_{T}(v|x_{&lt;t})=\frac{\exp(z_{v}/T)}{\sum_{v^{\prime}}\exp(z_{v^{\prime}}/T)}" display="block"><semantics><mrow><mrow><msub><mi>P</mi><mi>T</mi></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>z</mi><mi>v</mi></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mo>∑</mo><msup><mi>v</mi><mo>′</mo></msup></msub><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>z</mi><msup><mi>v</mi><mo>′</mo></msup></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math></div>

<ul><li><math alttext="T&lt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&lt;</mo><mn>1</mn></mrow></semantics></math>: Sharpens distribution <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> more deterministic, focused outputs.</li><li><math alttext="T=1" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>: Unmodified model distribution.</li><li><math alttext="T&gt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>: Flattens distribution <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> more random, creative outputs.</li><li><math alttext="T\to 0" display="inline"><semantics><mrow><mi>T</mi><mo stretchy="false">→</mo><mn>0</mn></mrow></semantics></math>: Becomes greedy decoding. <math alttext="T\to\infty" display="inline"><semantics><mrow><mi>T</mi><mo stretchy="false">→</mo><mi mathvariant="normal">∞</mi></mrow></semantics></math>: Becomes uniform sampling.</li></ul>

Common settings:<math alttext="T=0.7" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>0.7</mn></mrow></semantics></math> for factual tasks, <math alttext="T=1.0" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>1.0</mn></mrow></semantics></math>–<math alttext="1.2" display="inline"><semantics><mn>1.2</mn></semantics></math> for creative writing, <math alttext="T=0.0" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>0.0</mn></mrow></semantics></math> (greedy) for code/math.

### Contrastive Decoding

Contrastive decoding [211] exploits the difference between a strong model (expert) and a weak model (amateur) to amplify the expert’s unique knowledge:

<div class="hh-equation" id="ch1.ex50"><math alttext="x_{t}=\arg\max_{v\in\mathcal{V}(x_{&lt;t})}\left[\log P_{\text{expert}}(v|x_{&lt;t})-\log P_{\text{amateur}}(v|x_{&lt;t})\right]" display="block"><semantics><mrow><msub><mi>x</mi><mi>t</mi></msub><mo>=</mo><mrow><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><munder><mi>max</mi><mrow><mi>v</mi><mo>∈</mo><mrow><mi>𝒱</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></munder></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mtext>expert</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mtext>amateur</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math></div>

where <math alttext="\mathcal{V}(x_{&lt;t})=\{v:P_{\text{expert}}(v|x_{&lt;t})\geq\alpha\cdot\max_{v^{\prime}}P_{\text{expert}}(v^{\prime}|x_{&lt;t})\}" display="inline"><semantics><mrow><mrow><mi>𝒱</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo stretchy="false">{</mo><mi>v</mi><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mrow><msub><mi>P</mi><mtext>expert</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>≥</mo><mrow><mrow><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><msub><mi>max</mi><msup><mi>v</mi><mo>′</mo></msup></msub><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mtext>expert</mtext></msub></mrow></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>v</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math> is an adaptive plausibility constraint.

Intuition: The amateur model captures generic, obvious patterns (common words, repetition). Subtracting its log-probabilities removes this “generic signal,” leaving the expert’s distinctive knowledge and reasoning. Like removing background noise from a recording to hear the signal.

Pros: Reduces repetition and generic phrasing; improves factuality and coherence without additional training; works with any model pair. <br>

Cons: Requires running two models (2<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> compute); sensitive to amateur model choice; the plausibility threshold <math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math> needs tuning.

### Repetition Penalties

Orthogonal to the sampling strategy, repetition penalties discourage the model from repeating tokens. Given the raw logit <math alttext="z_{v}" display="inline"><semantics><msub><mi>z</mi><mi>v</mi></msub></semantics></math> for token <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math> (i.e., the unnormalized score output by the LM head <em>before</em> softmax), the penalized logit is:

<div class="hh-equation" id="ch1.ex51"><math alttext="z_{v}^{\prime}=\begin{cases}z_{v}/\theta&amp;\text{if }v\in\text{generated tokens and }z_{v}&gt;0\\
z_{v}\cdot\theta&amp;\text{if }v\in\text{generated tokens and }z_{v}&lt;0\end{cases}" display="block"><semantics><mrow><msubsup><mi>z</mi><mi>v</mi><mo>′</mo></msubsup><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><msub><mi>z</mi><mi>v</mi></msub><mo>/</mo><mi>θ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>v</mi></mrow><mo>∈</mo><mrow><mtext>generated tokens and </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>z</mi><mi>v</mi></msub></mrow><mo>&gt;</mo><mn>0</mn></mrow></mtd></mtr><mtr><mtd columnalign="left"><mrow><msub><mi>z</mi><mi>v</mi></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>θ</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>v</mi></mrow><mo>∈</mo><mrow><mtext>generated tokens and </mtext><mo lspace="0em" rspace="0em">​</mo><msub><mi>z</mi><mi>v</mi></msub></mrow><mo>&lt;</mo><mn>0</mn></mrow></mtd></mtr></mtable></mrow></mrow></semantics></math></div>

where <math alttext="\theta&gt;1" display="inline"><semantics><mrow><mi>θ</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math> is the penalty factor (typically 1.1–1.3). In both cases, the effect is to push the logit toward zero—reducing the probability of previously generated tokens. Frequency and presence penalties are simpler additive variants used by OpenAI APIs:

<div class="hh-equation" id="ch1.ex52"><math alttext="z_{v}^{\prime}=z_{v}-\alpha\cdot\text{count}(v)-\beta\cdot\mathbf{1}[v\in\text{generated}]" display="block"><semantics><mrow><msubsup><mi>z</mi><mi>v</mi><mo>′</mo></msubsup><mo>=</mo><msub><mi>z</mi><mi>v</mi></msub><mo>−</mo><mi>α</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mtext>count</mtext><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow><mo>−</mo><mi>β</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><mi>v</mi><mo>∈</mo><mtext>generated</mtext><mo stretchy="false">]</mo></mrow></mrow></semantics></math></div>

where <math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math> is the frequency penalty (proportional to how many times <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math> appeared) and <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> is the presence penalty (flat penalty for any prior occurrence).

### Practical Comparison

<div class="hh-table" id="ch1.t15"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Deterministic</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Diversity</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Quality</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Best For</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Greedy</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Yes</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>None</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Medium</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Code, factual QA</span></span></td></tr><tr><th class="ltx_align_left">Beam Search (<math alttext="B" display="inline"><semantics><mi>B</mi><span></span></semantics></math>=4–8)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Yes</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Low</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>High (narrow)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Translation, summarization</span></span></td></tr><tr><th class="ltx_align_left">Diverse Beam Search</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Yes</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Medium</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>High</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Candidate generation for reranking</span></span></td></tr><tr><th class="ltx_align_left">Top-<math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> (<math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math>=50)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Medium</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Medium</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>General-purpose generation</span></span></td></tr><tr><th class="ltx_align_left">Top-<math alttext="p" display="inline"><semantics><mi>p</mi><span></span></semantics></math> (<math alttext="p" display="inline"><semantics><mi>p</mi><span></span></semantics></math>=0.9)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Adaptive</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>High</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Default for open-ended tasks</span></span></td></tr><tr><th class="ltx_align_left">Min-<math alttext="p" display="inline"><semantics><mi>p</mi><span></span></semantics></math> (<math alttext="p_{\min}" display="inline"><semantics><msub><mi>p</mi><mi>min</mi></msub><span></span></semantics></math>=0.1)</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Adaptive</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>High</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Robust alternative to top-<math alttext="p" display="inline"><semantics><mi>p</mi><span></span></semantics></math></span></span></td></tr><tr><th class="ltx_align_left">Contrastive</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Yes</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Low</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Very High</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Factual, coherent long-form</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.15:</span>Decoding method comparison for LLM text generation.</p></div>

### Constrained Decoding (Structured Generation)

All methods above sample from the <em>full</em> vocabulary at each step. Constrained decoding restricts the set of allowed tokens so that the output is <em>guaranteed</em> to conform to a formal grammar—typically a JSON schema, regex, or context-free grammar (CFG).

At each decoding step <math alttext="t" display="inline"><semantics><mi>t</mi></semantics></math>, a token mask<math alttext="M_{t}\subseteq\mathcal{V}" display="inline"><semantics><mrow><msub><mi>M</mi><mi>t</mi></msub><mo>⊆</mo><mi>𝒱</mi></mrow></semantics></math> is computed from the current parser state. Only tokens in <math alttext="M_{t}" display="inline"><semantics><msub><mi>M</mi><mi>t</mi></msub></semantics></math> receive their original logits; all others are set to <math alttext="-\infty" display="inline"><semantics><mrow><mo>−</mo><mi mathvariant="normal">∞</mi></mrow></semantics></math> before softmax:

<div class="hh-equation" id="ch1.ex53"><math alttext="P^{\prime}(v|x_{&lt;t})=\begin{cases}P(v|x_{&lt;t})/Z&amp;\text{if }v\in M_{t}\\
0&amp;\text{otherwise}\end{cases}" display="block"><semantics><mrow><mrow><msup><mi>P</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mi>Z</mi></mrow></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mi>v</mi></mrow><mo>∈</mo><msub><mi>M</mi><mi>t</mi></msub></mrow></mtd></mtr><mtr><mtd columnalign="left"><mn>0</mn></mtd><mtd columnalign="left"><mtext>otherwise</mtext></mtd></mtr></mtable></mrow></mrow></semantics></math></div>

where <math alttext="Z=\sum_{v\in M_{t}}P(v|x_{&lt;t})" display="inline"><semantics><mrow><mi>Z</mi><mo rspace="0.111em">=</mo><mrow><msub><mo>∑</mo><mrow><mi>v</mi><mo>∈</mo><msub><mi>M</mi><mi>t</mi></msub></mrow></msub><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>x</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math> renormalizes. Because the mask changes every step (it depends on what has been generated so far), the constraint is enforced <em>incrementally</em>—the model cannot produce an invalid prefix at any point.

The compilation pipeline is:

<div class="hh-equation" id="ch1.ex54"><math alttext="\text{JSON Schema}\;\xrightarrow{\text{compile}}\;\text{Regex}\;\xrightarrow{\text{compile}}\;\text{FSM (DFA)}\;\xrightarrow{\text{index}}\;\text{Token Mask per State}" display="block"><semantics><mrow><mtext>JSON Schema</mtext><mover accent="true"><mo lspace="0.558em">→</mo><mtext mathsize="0.700em">compile</mtext></mover><mtext>Regex</mtext><mover accent="true"><mo lspace="0.558em">→</mo><mtext mathsize="0.700em">compile</mtext></mover><mtext>FSM (DFA)</mtext><mover accent="true"><mo lspace="0.558em">→</mo><mtext mathsize="0.700em">index</mtext></mover><mtext>Token Mask per State</mtext></mrow></semantics></math></div>

The FSM states correspond to positions in the regex. For each state, all vocabulary tokens that would keep the string in the language are precomputed into an index (a one-time cost per schema). At runtime, looking up the mask is an <math alttext="O(1)" display="inline"><semantics><mrow><mi>O</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></mrow></semantics></math> table access—adding negligible latency to each decoding step.

<ul><li>Outlines[378]: Compiles JSON schemas and regexes into interleaved FSM-guided generation. Supports any model with a logits interface.</li><li>lm-format-enforcer: Similar FSM approach with a focus on integration with serving frameworks (vLLM, TGI).</li><li>Guidance (Microsoft): Interleaves constrained generation with control flow (loops, conditions), enabling complex structured outputs beyond flat schemas.</li><li>XGrammar[77]: Pushdown-automaton-based engine supporting full context-free grammars (not just regular languages), used in MLC-LLM and vLLM for grammar-mode decoding.</li></ul>

Constrained decoding <em>guarantees</em> syntactic validity—no post-hoc parsing failures, no retries. However:

<ul><li>Semantic quality: Forcing structure can degrade content quality if the model’s probability mass for the “correct” answer lies outside the grammar. In practice this is rare for well-trained models on well-designed schemas.</li><li>Compilation cost: The FSM index must be built per schema. For complex schemas this can take 1–5 s, but it is amortized over all requests using that schema.</li><li>Grammar coverage: Regex/FSM handles JSON, YAML, SQL fragments, and most structured formats. Full CFGs (via XGrammar or LALR parsers) cover languages like Python or XML.</li></ul>

## Prompt Engineering

Prompt engineering is the discipline of designing inputs to LLMs that reliably elicit desired behaviour—without changing model weights. While fine-tuning modifies the model, prompt engineering exploits the model’s <em>existing</em> capabilities through careful framing, examples, and structure. It is the fastest, cheapest, and most accessible lever for improving LLM outputs, and remains essential even when using fine-tuned models.

### In-Context Learning (ICL)

In-context learning [32] is the remarkable ability of large language models to learn tasks at inference time purely from examples provided in the prompt—with no gradient updates. The model implicitly infers the task from the pattern of input–output pairs and generalizes to new inputs.

ICL emerges primarily in models above <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>1B parameters and improves log-linearly with model scale [32]. Smaller models can memorize examples but struggle to generalize to novel inputs within the same context window.

### Zero-Shot Prompting

Zero-shot prompting provides <em>no</em> examples—only a task description or instruction. The model must rely entirely on its pretrained knowledge and instruction-tuning to produce the correct format and content.

<ul><li>Tasks the model has seen extensively during pretraining/SFT (translation, summarization, sentiment)</li><li>Well-specified instructions with unambiguous output format</li><li>Instruction-tuned models (e.g., ChatGPT, Claude, Llama-3-Instruct) significantly outperform base models at zero-shot tasks [280]</li></ul>

Novel formats, domain-specific labeling schemes, or ambiguous tasks where the model cannot infer your exact requirements from the instruction alone.

### Few-Shot Prompting

Few-shot prompting [32] provides <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> input–output examples (“shots”) before the actual query. This is the most common form of in-context learning and remains one of the most effective prompting strategies.

<ol><li>Diversity: Cover the range of expected inputs (different lengths, edge cases, categories).</li><li>Ordering: Place harder or more representative examples last (recency bias) [242].</li><li>Label balance: If classifying, include examples from all classes to avoid majority-class bias.</li><li>Format consistency: Every example must follow the <em>exact</em> same structure. The model mimics the pattern.</li><li>Relevance: Use examples semantically similar to the target query for best results [224].</li></ol>

Performance typically improves from 0 to 4–8 examples, then plateaus. Beyond <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>20 examples, gains are marginal and you risk filling the context window. Min et al. [257] showed that the <em>format</em> and <em>label space</em> of examples matter more than label correctness—even random labels help (though correct labels help more).

### Instruction-Following Prompts

Instruction-tuned models respond best to clear, structured instructions. The key insight: treat the prompt as a <em>specification</em>, not a suggestion.

Modern chat APIs separate the <em>system</em> prompt (persistent instructions, role definition) from the <em>user</em> message (per-turn input). System prompts are processed with higher attention priority in most models and provide a natural place for role definitions, constraints, and output format specifications [274].

### Structured Output Prompts (JSON/XML)

For programmatic use, the most critical prompting technique is enforcing structured output—particularly JSON.

<ul><li>Schema-first: Show the exact JSON schema <em>before</em> the input. The model treats it as a template.</li><li>Constrained decoding: Use grammar-based sampling (e.g., Outlines [378], Guidance) to guarantee syntactically valid JSON at the token level.</li><li>XML tags: For nested or multi-part outputs, XML tags (e.g., <code>&lt;thinking&gt;...&lt;/thinking&gt;</code>) provide unambiguous delimiters that models follow reliably.</li><li>Pydantic/TypeScript types: Providing type definitions helps models understand field constraints (OpenAI’s function calling uses JSON Schema internally).</li></ul>

A distinct but complementary technique is <em>JSON prompting</em>—formatting the prompt <em>itself</em> as JSON rather than natural language. This exploits the model’s extensive pre-training on structured data (APIs, configs, code) to improve instruction adherence, reduce ambiguity, and enable deterministic parsing of multi-field requests.

### Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting [374] asks the model to produce intermediate reasoning steps before giving a final answer. This simple technique dramatically improves performance on tasks requiring multi-step reasoning: arithmetic, logic, commonsense inference, and code generation.

<ul><li>Serializes computation: Transformers have fixed depth but variable-length generation. CoT converts parallel (hard) problems into sequential (easy) steps, effectively increasing the model’s computational budget.</li><li>Reduces compounding errors: Each step is a simpler sub-problem with lower per-step error rate.</li><li>Exposes intermediate state: Makes reasoning auditable and debuggable.</li></ul>

Wang et al. [367] showed that sampling multiple chain-of-thought reasoning paths and taking a majority vote over final answers significantly outperforms single-path CoT. The intuition: correct reasoning paths tend to converge on the same answer, while errors are typically idiosyncratic. This trades compute (generating <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> samples) for accuracy—practical when latency is less important than correctness.

CoT is not universally beneficial. For simple tasks (single-step classification, retrieval, formatting), CoT adds unnecessary tokens, increases latency, and can even introduce errors through overthinking. Use CoT selectively for tasks where you expect multi-step reasoning to be required.

### Advanced Prompting Techniques

Rather than relying solely on the model’s parametric memory, RAG [209] retrieves relevant documents and includes them in the prompt:

```python
Context (retrieved): [document chunks]
Question: [user query]
Answer based ONLY on the provided context.
```

This grounds the model’s responses in verifiable sources and dramatically reduces hallucinations for knowledge-intensive tasks.

Complex tasks benefit from being broken into a pipeline of simpler prompts, where the output of one becomes the input to the next:

<ol><li><em>Extract</em> key facts from document</li><li><em>Reason</em> over extracted facts</li><li><em>Format</em> final answer</li></ol>

Each step can use a different prompt template, model, or temperature setting. This is more controllable than a single monolithic prompt and enables targeted debugging.

Bai et al. [16] introduce prompts that ask the model to critique and revise its own output against a set of principles:

```python
[Generate initial response]
Critique: Does this response violate any of the following
principles? [list principles]
Revision: Rewrite the response addressing the critique.
```

Rather than hand-crafting prompts, recent work automates prompt design:

<ul><li>APE[441]: Uses an LLM to generate and score candidate prompts automatically.</li><li>DSPy[182]: Compiles declarative task descriptions into optimized prompt pipelines with learned few-shot examples.</li><li>OPRO[402]: Treats prompt optimization as an optimization problem, using an LLM as the optimizer.</li></ul>

ARQ [403] addresses a fundamental weakness of standard prompting: as context length grows, models increasingly “lose” critical information in the middle of the prompt (the <em>lost-in-the-middle</em> effect). ARQ mitigates this by decomposing a complex query into multiple focused sub-queries, each designed to direct the model’s attention to a specific part of the context:

<ol><li>Query decomposition: Break the user question into atomic sub-questions that each target a narrow aspect.</li><li>Attentive retrieval: For each sub-query, retrieve or highlight only the relevant context slice—forcing the model to attend to it.</li><li>Aggregation: Combine sub-answers into a coherent final response.</li></ol>

This is particularly effective for long-document QA, multi-hop reasoning over large retrieval sets, and agentic tasks where the context window contains many tool outputs. ARQ can be seen as a structured form of chain-of-thought that explicitly manages <em>where</em> the model looks, not just <em>how</em> it reasons.

### Best Practices: Crafting Effective Prompts

Based on empirical findings across the literature and practitioner experience, the following principles reliably improve prompt quality:

<div class="hh-table" id="ch1.t16"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Failure Mode</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Symptom</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Solution</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">Instruction amnesia</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Model ignores constraints in long prompts</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Move constraints to end; repeat key rules; use system prompt</span></span></td></tr><tr><th class="ltx_align_left">Format drift</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Output starts correct but degrades over long generations</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Use constrained decoding; break into shorter chained prompts</span></span></td></tr><tr><th class="ltx_align_left">Sycophancy</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Model agrees with incorrect premises in the prompt</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Add “challenge assumptions if incorrect”; use system-level instruction</span></span></td></tr><tr><th class="ltx_align_left">Hallucinated details</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Model invents facts not in provided context</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Add “if unknown, say I don’t know”; use RAG with source attribution</span></span></td></tr><tr><th class="ltx_align_left">Refusal over-triggering</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Model refuses benign requests due to safety training</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Rephrase to clarify legitimate intent; provide explicit context for why the request is appropriate</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.16:</span>Common prompting failure modes and solutions.</p></div>

## Model Compression Methods

Model compression reduces model size and inference cost while preserving quality. Three main approaches: quantization (reduce precision), pruning (remove parameters), and distillation (train a smaller model to mimic a larger one).

### Quantization

Quantization reduces model size and inference cost by representing weights (and optionally activations) in lower-precision formats. The core trade-off is compression ratio versus quality degradation.

<div class="hh-table" id="ch1.t17"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left"><span>Bits</span></th><th class="ltx_align_left"><span>Type</span></th><th class="ltx_align_left"><span>Key Idea</span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>GPTQ</span><span>[99]</span></td><td class="ltx_align_left">4-bit</td><td class="ltx_align_left">PTQ, weight-only</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Layer-wise quantization minimizing <math alttext="\|WX-\hat{W}X\|^{2}" display="inline"><semantics><msup><mrow><mo stretchy="false">‖</mo><mrow><mrow><mi>W</mi><mo lspace="0em" rspace="0em">​</mo><mi>X</mi></mrow><mo>−</mo><mrow><mover accent="true"><mi>W</mi><mo>^</mo></mover><mo lspace="0em" rspace="0em">​</mo><mi>X</mi></mrow></mrow><mo stretchy="false">‖</mo></mrow><mn>2</mn></msup><span></span></semantics></math> via optimal brain surgeon.</span></span></td></tr><tr><td class="ltx_align_left"><span>AWQ</span><span>[218]</span></td><td class="ltx_align_left">4-bit</td><td class="ltx_align_left">PTQ, weight-only</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Protects salient weights (those with large activations). 1% of weights carry 99% importance.</span></span></td></tr><tr><td class="ltx_align_left"><span>GGUF</span><span>[109]</span></td><td class="ltx_align_left">2–8 bit</td><td class="ltx_align_left">PTQ, weight-only</td><td class="ltx_align_left"><span class="ltx_align_top"><span>CPU-optimized format (llama.cpp). Per-block quantization with multiple types.</span></span></td></tr><tr><td class="ltx_align_left"><span>FP8</span> (E4M3)</td><td class="ltx_align_left">8-bit</td><td class="ltx_align_left">Training + inference</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Native H100 support. 2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math> throughput vs BF16.</span></span></td></tr><tr><td class="ltx_align_left"><span>SmoothQuant</span><span>[392]</span></td><td class="ltx_align_left">W8A8</td><td class="ltx_align_left">PTQ, weight+act.</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Smooths activation outliers into weights before quantization. Enables INT8 GEMM.</span></span></td></tr><tr><td class="ltx_align_left"><span>QAT</span><span>[235]</span></td><td class="ltx_align_left">4-bit</td><td class="ltx_align_left">QAT</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Trains with simulated quantization. Highest quality but expensive.</span></span></td></tr><tr><td class="ltx_align_left"><span>AQLM</span><span>[83]</span></td><td class="ltx_align_left">2-bit</td><td class="ltx_align_left">PTQ, additive codes</td><td class="ltx_align_left"><span class="ltx_align_top"><span>Extreme compression via learned additive quantization codebooks.</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.17:</span>Quantization methods for LLMs.</p></div>

### Pruning

Modern LLMs contain billions of parameters, yet empirical studies consistently show that a large fraction of these weights contribute minimally to model outputs. Pruning exploits this over-parameterization: by removing redundant weights, we reduce memory footprint (enabling deployment on smaller GPUs or edge devices), inference latency (fewer multiply-accumulate operations per forward pass), and serving cost (higher throughput per dollar). Unlike quantization, which reduces the precision of all weights uniformly, pruning selectively eliminates weights—enabling multiplicative savings when combined with quantization (e.g., a 50% sparse, 4-bit model uses <math alttext="4\times" display="inline"><semantics><mrow><mn>4</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> less memory than the dense BF16 baseline). The challenge is achieving high sparsity without degrading generation quality, which has driven the development of principled one-shot methods that require no retraining.

### Knowledge Distillation

Knowledge distillation [139] transfers the learned behaviour of a large <em>teacher</em> model into a smaller, cheaper <em>student</em> model. The core idea is that the teacher’s output distribution over tokens carries far richer signal than ground-truth hard labels alone — revealing inter-class similarities, calibration, and uncertainty that the student can exploit.

To expose the “dark knowledge” in the teacher’s logits we soften the distribution with a temperature <math alttext="T&gt;1" display="inline"><semantics><mrow><mi>T</mi><mo>&gt;</mo><mn>1</mn></mrow></semantics></math>:

<div class="hh-equation" id="ch1.ex56"><math alttext="p_{i}^{(T)}=\frac{\exp(z_{i}/T)}{\sum_{j}\exp(z_{j}/T)}" display="block"><semantics><mrow><msubsup><mi>p</mi><mi>i</mi><mrow><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow></msubsup><mo>=</mo><mfrac><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>z</mi><mi>i</mi></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mrow><msub><mo>∑</mo><mi>j</mi></msub><mrow><mi>exp</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>z</mi><mi>j</mi></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math></div>

At high temperature the probability mass spreads across more tokens, making near-miss alternatives visible. During training the same temperature is applied to the student; at inference the student uses <math alttext="T=1" display="inline"><semantics><mrow><mi>T</mi><mo>=</mo><mn>1</mn></mrow></semantics></math>.

<div class="hh-equation" id="ch1.ex57"><math alttext="\mathcal{L}_{\text{distill}}=\alpha\,T^{2}\cdot\text{KL}\!\bigl(P_{\text{teacher}}^{(T)}\;\|\;P_{\text{student}}^{(T)}\bigr)\;+\;(1-\alpha)\cdot\mathcal{L}_{\text{CE}}(y,P_{\text{student}}^{(1)})" display="block"><semantics><mrow><msub><mi>ℒ</mi><mtext>distill</mtext></msub><mo>=</mo><mi>α</mi><msup><mi>T</mi><mn>2</mn></msup><mo lspace="0.222em" rspace="0.222em">⋅</mo><mpadded width="1.431em"><mtext>KL</mtext></mpadded><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><msubsup><mi>P</mi><mtext>teacher</mtext><mrow><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow></msubsup><mo lspace="0em" rspace="0.447em">∥</mo><msubsup><mi>P</mi><mtext>student</mtext><mrow><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mrow></msubsup><mo maxsize="1.200em" minsize="1.200em" rspace="0.280em">)</mo></mrow><mo rspace="0.502em">+</mo><mrow><mo stretchy="false">(</mo><mn>1</mn><mo>−</mo><mi>α</mi><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><msub><mi>ℒ</mi><mtext>CE</mtext></msub><mrow><mo stretchy="false">(</mo><mi>y</mi><mo>,</mo><msubsup><mi>P</mi><mtext>student</mtext><mrow><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo></mrow></msubsup><mo stretchy="false">)</mo></mrow></mrow></semantics></math></div>

The <math alttext="T^{2}" display="inline"><semantics><msup><mi>T</mi><mn>2</mn></msup></semantics></math> factor compensates for the reduced gradient magnitude of softened distributions. Typical values: <math alttext="T\in[2,20]" display="inline"><semantics><mrow><mi>T</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>2</mn><mo>,</mo><mn>20</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>, <math alttext="\alpha\in[0.5,0.9]" display="inline"><semantics><mrow><mi>α</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.5</mn><mo>,</mo><mn>0.9</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> (more weight on KL when teacher quality is high).

<div class="hh-table" id="ch1.t18"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Paradigm</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Mechanism</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Pros</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Cons</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Offline / White-box</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Teacher logits pre-computed; student trains on full distributions</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Full distribution signal; one-time teacher cost</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Stale data; storage heavy</span></span></td></tr><tr><th class="ltx_align_left"><span>Online / Co-training</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Teacher generates on-the-fly; student sees fresh logits</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Adapts to student weaknesses</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="2\times" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0.222em">×</mo></mrow><span></span></semantics></math> compute; complex infra</span></span></td></tr><tr><th class="ltx_align_left"><span>Black-box (API)</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Only teacher <em>text</em> outputs available (no logits)</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Works with proprietary models</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Loses dark knowledge; SFT-like</span></span></td></tr><tr><th class="ltx_align_left"><span>Self-distillation</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Model distills into a smaller version of itself</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>No separate teacher needed</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Teacher = student family; ceiling</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.18:</span>Knowledge distillation paradigms for LLMs.</p></div>

The teacher’s full logit vector (or top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> logits for storage efficiency) is recorded for each training token. The student minimises the KL divergence against these stored distributions. This is the most data-efficient paradigm when teacher access is unrestricted.

Motivation: Decouple teacher inference from student training — run the teacher once on high-end hardware, then train many students cheaply.

Pros: Deterministic, reproducible; teacher cost is amortised; full distributional signal. <br>

Cons: Requires storing <math alttext="|V|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><mi>V</mi><mo stretchy="false">|</mo></mrow></semantics></math>-dimensional vectors per token (mitigated by top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> pruning); teacher cannot adapt to student failures.

Teacher and student are run jointly: the teacher generates logits for the student’s current training batch.

Motivation: Let the teacher focus on inputs where the student currently struggles (curriculum-like).

Pros: Freshness; can use student-generated inputs for on-policy distillation. <br>

Cons: Double the GPU cost; synchronisation complexity; harder to scale.

When only text outputs are available (e.g. distilling from a proprietary API), the student is trained via SFT on the teacher’s generations, optionally augmented with chain-of-thought traces.

Motivation: Practical reality — most frontier models do not expose logits.

Pros: Simple pipeline; works with any model behind an API. <br>

Cons: No soft-label signal; prone to hallucination amplification; effectively supervised fine-tuning.

A model distils from a larger version within the same architecture family (e.g. Llama-3 70B <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> 8B) or from its own checkpoints during training.

Motivation: Avoid training a separate teacher; leverage the model’s own capacity at different scales.

Pros: Architecture compatibility; no external dependency. <br>

Cons: Teacher ceiling equals model ceiling; cannot introduce genuinely new knowledge.

<ul><li>Sequence-level vs. token-level: Token-level KL is standard; sequence-level distillation (minimising KL over full sequences) better captures long-range coherence but is harder to optimise.</li><li>Layer-wise hints: Matching intermediate representations (attention maps, hidden states) provides additional learning signal — especially useful when student architecture differs.</li><li>Data selection: Distillation data quality matters; curating diverse, hard examples yields better students than random sampling.</li><li>Student capacity: Diminishing returns below <math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>10% of teacher parameters; at extreme compression, architecture changes (e.g. MoE <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> dense) may be needed.</li><li>Combining with quantization: Distillation + 4-bit quantization (e.g. QLoRA-distilled models) achieves near-teacher quality at <math alttext="20\times" display="inline"><semantics><mrow><mn>20</mn><mo lspace="0.222em">×</mo></mrow></semantics></math> compression.</li></ul>

## Speculative Decoding Methods

Speculative decoding [206] accelerates autoregressive generation by predicting multiple tokens simultaneously, then verifying them in a single forward pass of the target model. It produces identical output distribution to standard decoding (no quality loss) while achieving 2–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> speedup.

### Core Principle

### Methods Comparison

<div class="hh-table" id="ch1.t19"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left"><span>Draft Source</span></th><th class="ltx_align_left"><span>Speedup</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Key Idea</span></span></span></th></tr></thead><tbody><tr><td class="ltx_align_left"><span>Standard</span><span>[206]</span></td><td class="ltx_align_left">Small model (1–7B)</td><td class="ltx_align_left">2–3<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Separate draft model generates candidates. Simple but requires loading 2 models.</span></span></td></tr><tr><td class="ltx_align_left"><span>Medusa</span><span>[35]</span></td><td class="ltx_align_left">Parallel LM heads</td><td class="ltx_align_left">2–3<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Add <math alttext="k" display="inline"><semantics><mi>k</mi><span></span></semantics></math> extra prediction heads to the target model. Each predicts token at position <math alttext="+1,+2,\ldots,+k" display="inline"><semantics><mrow><mrow><mo>+</mo><mn>1</mn></mrow><mo>,</mo><mrow><mo>+</mo><mn>2</mn></mrow><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><mo>+</mo><mi>k</mi></mrow></mrow><span></span></semantics></math>.</span></span></td></tr><tr><td class="ltx_align_left"><span>Eagle</span><span>[213]</span></td><td class="ltx_align_left">Feature-level</td><td class="ltx_align_left">2.5–3.5<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Lightweight decoder generates draft tokens from target model’s hidden states. Higher acceptance than Medusa.</span></span></td></tr><tr><td class="ltx_align_left"><span>Eagle-2</span><span>[213]</span></td><td class="ltx_align_left">Context-aware</td><td class="ltx_align_left">3–4<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Dynamic draft tree with confidence-based expansion. State-of-the-art acceptance rates.</span></span></td></tr><tr><td class="ltx_align_left"><span>N-gram Lookup</span></td><td class="ltx_align_left">N-gram cache</td><td class="ltx_align_left">1.5–2<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Match prompt n-grams against previously generated text. Zero cost; great for repetitive outputs.</span></span></td></tr><tr><td class="ltx_align_left"><span>Lookahead</span><span>[101]</span></td><td class="ltx_align_left">Jacobi iteration</td><td class="ltx_align_left">2–2.5<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Parallel Jacobi decoding with n-gram verification. No draft model; uses target model itself.</span></span></td></tr><tr><td class="ltx_align_left"><span>Multi-token</span><span>[111]</span></td><td class="ltx_align_left">Modified arch.</td><td class="ltx_align_left">2–3<math alttext="\times" display="inline"><semantics><mo>×</mo><span></span></semantics></math></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Train the model to natively predict multiple tokens per step (Meta’s approach in Llama).</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.19:</span>Speculative decoding methods supported by modern inference engines.</p></div>

### Medusa: Multi-Head Speculative Decoding

### Eagle: Feature-Level Drafting

### N-gram Speculative Decoding

### Integration with vLLM

```python
from vllm import LLM, SamplingParams

# Standard speculative decoding (separate draft model)
llm = LLM(
    model="meta-llama/Llama-3-70B",
    tensor_parallel_size=4,
    speculative_config={
        "model": "meta-llama/Llama-3-8B",
        "num_speculative_tokens": 5,
    },
)

# N-gram speculation (zero-cost, no draft model needed)
llm = LLM(
    model="meta-llama/Llama-3-70B",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,  # Match up to 4-grams from prompt
    },
)

# EAGLE-style (feature-level draft, high acceptance rate)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "num_speculative_tokens": 2,
        "method": "eagle",
        "draft_tensor_parallel_size": 1,
    },
)

# MLP speculator (IBM-style, lightweight head)
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        "draft_tensor_parallel_size": 1,
    },
)
```

## Hallucination Detection

LLMs generate fluent text that may be factually incorrect—a phenomenon called hallucination[164]. This section covers basic detection methods at the model level (without external retrieval or multi-agent verification).

### Types of Hallucination

### Detection Methods (Model-Level)

<div class="hh-table" id="ch1.t20"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Method</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Mechanism</span></span></span></th><th class="ltx_align_left"><span>Signal</span></th></tr></thead><tbody><tr><td class="ltx_align_left">Token-level entropy</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>High entropy at generation time indicates uncertainty <span>[175]</span></span></span></td><td class="ltx_align_left"><math alttext="H(P(x_{t}))&gt;\tau" display="inline"><semantics><mrow><mrow><mi>H</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>&gt;</mo><mi>τ</mi></mrow><span></span></semantics></math></td></tr><tr><td class="ltx_align_left">Sequence log-prob</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Low average log-probability of the output suggests confabulation</span></span></td><td class="ltx_align_left"><math alttext="\frac{1}{T}\sum_{t}\log P(x_{t})" display="inline"><semantics><mrow><mfrac><mn>1</mn><mi>T</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><msub><mo>∑</mo><mi>t</mi></msub><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><span></span></semantics></math></td></tr><tr><td class="ltx_align_left">Consistency sampling</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Generate <math alttext="N" display="inline"><semantics><mi>N</mi><span></span></semantics></math> responses; low agreement <math alttext="=" display="inline"><semantics><mo>=</mo><span></span></semantics></math> likely hallucination <span>[249]</span></span></span></td><td class="ltx_align_left">Contradiction rate</td></tr><tr><td class="ltx_align_left">Semantic entropy</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Cluster meanings (not strings); high semantic entropy <math alttext="=" display="inline"><semantics><mo>=</mo><span></span></semantics></math> uncertain <span>[193]</span></span></span></td><td class="ltx_align_left">Cluster diversity</td></tr><tr><td class="ltx_align_left">DoLA</td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Contrast logits between later vs. earlier layers; amplifies factual knowledge <span>[58]</span></span></span></td><td class="ltx_align_left">Layer divergence</td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.20:</span>Basic hallucination detection methods that operate at the model level.</p></div>

Kuhn et al. [193] observe that token-level entropy is unreliable (paraphrases have different tokens but same meaning). Instead, they generate multiple responses, cluster them by semantic equivalence (via NLI), and compute entropy over meaning clusters:

<div class="hh-equation" id="ch1.ex58"><math alttext="SE=-\sum_{c\in\text{clusters}}P(c)\log P(c)" display="block"><semantics><mrow><mi>S</mi><mi>E</mi><mo rspace="0em">=</mo><mo lspace="0em" rspace="0.055em">−</mo><munder><mo movablelimits="false">∑</mo><mrow><mi>c</mi><mo>∈</mo><mtext>clusters</mtext></mrow></munder><mi>P</mi><mrow><mo stretchy="false">(</mo><mi>c</mi><mo rspace="0.167em" stretchy="false">)</mo></mrow><mi>log</mi><mi>P</mi><mrow><mo stretchy="false">(</mo><mi>c</mi><mo stretchy="false">)</mo></mrow></mrow></semantics></math></div>

High SE means the model produces <em>semantically different</em> answers—a strong hallucination signal.

Manakul et al. [249] detect hallucinations by checking self-consistency: generate multiple responses and verify whether claims in the main response are supported by the alternatives. If the model “disagrees with itself,” the claim is likely hallucinated. No external knowledge needed.

Chuang et al. [58] observe that factual knowledge emerges in later transformer layers while earlier layers retain more generic/uncertain representations. DoLA contrasts the logit distributions between a later (“mature”) layer and an earlier (“premature”) layer at each decoding step:

<div class="hh-equation" id="ch1.ex59"><math alttext="\text{DoLA}(x_{t})=\text{softmax}\!\bigl(\log P_{\text{late}}(x_{t})-\log P_{\text{early}}(x_{t})\bigr)" display="block"><semantics><mrow><mrow><mtext>DoLA</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="3.720em"><mtext>softmax</mtext></mpadded><mo lspace="0em" rspace="0em">​</mo><mrow><mo maxsize="1.200em" minsize="1.200em">(</mo><mrow><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mtext>late</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mtext>early</mtext></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>t</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo maxsize="1.200em" minsize="1.200em">)</mo></mrow></mrow></mrow></semantics></math></div>

By amplifying the signal from factual knowledge encoded in deeper layers, DoLA reduces hallucinations at inference time <em>without any retraining</em>—requiring only a single additional forward pass through the contrasted layer. It is complementary to sampling-based methods and can be combined with them.

## LLM Safety and Responsible AI

Safety is not an afterthought—it is an integral part of the LLM training pipeline. This section covers the key dimensions of LLM safety and the mechanisms used to enforce responsible behavior.

### Threat Taxonomy

<div class="hh-table" id="ch1.t21"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>Category</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>Description and Examples</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left"><span>Harmful content</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Generating toxic, violent, or illegal instructions (bioweapons, CSAM)</span></span></td></tr><tr><th class="ltx_align_left"><span>Bias and discrimination</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Perpetuating stereotypes; unfair treatment across demographics <span>[104]</span></span></span></td></tr><tr><th class="ltx_align_left"><span>Privacy violations</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Leaking PII from training data; memorization attacks <span>[39]</span></span></span></td></tr><tr><th class="ltx_align_left"><span>Jailbreaking</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Adversarial prompts that bypass safety guardrails <span>[443]</span></span></span></td></tr><tr><th class="ltx_align_left"><span>Misinformation</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Generating convincing but false claims (hallucination at scale)</span></span></td></tr><tr><th class="ltx_align_left"><span>Dual-use</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>Legitimate capabilities (coding, chemistry) weaponized for harm</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">Table 1.21:</span>LLM safety threat categories.</p></div>

### Safety Training Pipeline

<figure id="ch1.f14"><img src="./fig_015_fig15.png" alt="Figure 1.14: Safety is applied at every stage: data filtering in pretraining, refusal examples in SFT, safety-specific r" loading="lazy" decoding="async"><figcaption><span class="hh-tag">Figure 1.14:</span>Safety is applied at every stage: data filtering in pretraining, refusal examples in SFT, safety-specific reward models in RLHF, and iterative red-teaming.</figcaption></figure>

### Key Safety Mechanisms

### The Helpfulness–Safety Tradeoff

### Evaluation

<ul><li>Safety benchmarks: ToxiGen, RealToxicityPrompts, BBQ (bias), CrowS-Pairs</li><li>Jailbreak robustness: GCG attacks [443], multi-turn jailbreaks, encoded prompts</li><li>Over-refusal rate: Measure false-positive refusals on benign prompts (target <math alttext="&lt;" display="inline"><semantics><mo>&lt;</mo></semantics></math>5%)</li><li>Red team evaluations: Human adversarial testing with domain experts (biosecurity, cybersecurity)</li></ul>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">Chapter 2 Systems Foundations for LLMs</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>

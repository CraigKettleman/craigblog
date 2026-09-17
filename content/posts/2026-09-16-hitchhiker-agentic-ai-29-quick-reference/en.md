---
title: "The Hitchhiker's Guide to Agentic AI · Chapter 29 Quick Reference"
slug: "hitchhiker-agentic-ai-29-quick-reference"
lang: "en"
date: "2026-09-16T00:30:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "This chapter consolidates key equations, architecture specifications, API references, and failure mode diagnostics for rapid lookup during d…"
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
<p class="hh-part-title">Part VI Assessment & Reference</p>
</aside>

This chapter consolidates key equations, architecture specifications, API references, and failure mode diagnostics for rapid lookup during development and debugging.

## Core RL & Alignment Equations



## Transformer & Architecture Formulas



## Decoding Methods

MethodFormula / RuleKey ParamGreedy<math alttext="y_{t}=\arg\max_{v}P(v|y_{&lt;t})" display="inline"><semantics><mrow><msub><mi>y</mi><mi>t</mi></msub><mo>=</mo><mrow><mi>arg</mi><mo lspace="0.167em">⁡</mo><mrow><msub><mi>max</mi><mi>v</mi></msub><mo lspace="0.167em">⁡</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo lspace="0em" rspace="0em">|</mo><msub><mi>y</mi><mrow><mphantom></mphantom><mo>&lt;</mo><mi>t</mi></mrow></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math>—Beam searchKeep top-<math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> partial sequences by joint probability<math alttext="B=4" display="inline"><semantics><mrow><mi>B</mi><mo>=</mo><mn>4</mn></mrow></semantics></math>–<math alttext="8" display="inline"><semantics><mn>8</mn></semantics></math>Temperature<math alttext="P^{\prime}(v)=\text{softmax}(\text{logit}_{v}/T)" display="inline"><semantics><mrow><mrow><msup><mi>P</mi><mo>′</mo></msup><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mtext>softmax</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo>/</mo><mi>T</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><math alttext="T\in[0.1,1.5]" display="inline"><semantics><mrow><mi>T</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0.1</mn><mo>,</mo><mn>1.5</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math>Top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>Zero out all but top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> logits, renormalize<math alttext="k=40" display="inline"><semantics><mrow><mi>k</mi><mo>=</mo><mn>40</mn></mrow></semantics></math>–<math alttext="100" display="inline"><semantics><mn>100</mn></semantics></math>Top-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math> (nucleus)Keep smallest set <math alttext="V^{\prime}" display="inline"><semantics><msup><mi>V</mi><mo>′</mo></msup></semantics></math> s.t. <math alttext="\sum_{v\in V^{\prime}}P(v)\geq p" display="inline"><semantics><mrow><mrow><msub><mo>∑</mo><mrow><mi>v</mi><mo>∈</mo><msup><mi>V</mi><mo>′</mo></msup></mrow></msub><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>≥</mo><mi>p</mi></mrow></semantics></math><math alttext="p=0.9" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>0.9</mn></mrow></semantics></math>–<math alttext="0.95" display="inline"><semantics><mn>0.95</mn></semantics></math>Min-<math alttext="p" display="inline"><semantics><mi>p</mi></semantics></math>Keep tokens with <math alttext="P(v)\geq p_{\text{min}}\cdot P(v_{\text{max}})" display="inline"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>v</mi><mo stretchy="false">)</mo></mrow></mrow><mo>≥</mo><mrow><msub><mi>p</mi><mtext>min</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>v</mi><mtext>max</mtext></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math><math alttext="p_{\text{min}}=0.05" display="inline"><semantics><mrow><msub><mi>p</mi><mtext>min</mtext></msub><mo>=</mo><mn>0.05</mn></mrow></semantics></math>–<math alttext="0.1" display="inline"><semantics><mn>0.1</mn></semantics></math>Repetition penalty<math alttext="\text{logit}_{v}\leftarrow\text{logit}_{v}/\theta" display="inline"><semantics><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo stretchy="false">←</mo><mrow><msub><mtext>logit</mtext><mi>v</mi></msub><mo>/</mo><mi>θ</mi></mrow></mrow></semantics></math> if <math alttext="v" display="inline"><semantics><mi>v</mi></semantics></math> appeared before<math alttext="\theta=1.1" display="inline"><semantics><mrow><mi>θ</mi><mo>=</mo><mn>1.1</mn></mrow></semantics></math>–<math alttext="1.3" display="inline"><semantics><mn>1.3</mn></semantics></math>

## Systems & Parallelism

FormulaValue (70B, BF16)DescriptionModel memory<math alttext="2P" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow></semantics></math> bytes<math alttext="140" display="inline"><semantics><mn>140</mn></semantics></math> GB (weights only)Adam optimizer<math alttext="2P\times 4" display="inline"><semantics><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo lspace="0.222em" rspace="0.222em">×</mo><mn>4</mn></mrow></semantics></math> bytes (m + v)<math alttext="280" display="inline"><semantics><mn>280</mn></semantics></math> GBFull training footprint<math alttext="\sim 8P" display="inline"><semantics><mrow><mphantom></mphantom><mo>∼</mo><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow></mrow></semantics></math> bytes<math alttext="560" display="inline"><semantics><mn>560</mn></semantics></math> GB (weights + opt + grad)FSDP memory/GPU<math alttext="8P/N_{\text{GPUs}}" display="inline"><semantics><mrow><mrow><mn>8</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>/</mo><msub><mi>N</mi><mtext>GPUs</mtext></msub></mrow></semantics></math><math alttext="70" display="inline"><semantics><mn>70</mn></semantics></math> GB with 8 GPUsGen arithmetic intensity<math alttext="2P/2P=1" display="inline"><semantics><mrow><mrow><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>/</mo><mn>2</mn></mrow><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>=</mo><mn>1</mn></mrow></semantics></math> FLOP/byteHeavily memory-boundToken rate (gen)HBM_BW <math alttext="/(2P)" display="inline"><semantics><mrow><mphantom></mphantom><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math><math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>14 tok/s (A100, batch=1)TP AllReduce / layer<math alttext="2\times 2\cdot\frac{T-1}{T}\cdot bsd" display="inline"><semantics><mrow><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">×</mo><mn>2</mn></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><mrow><mi>T</mi><mo>−</mo><mn>1</mn></mrow><mi>T</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mi>b</mi><mo lspace="0em" rspace="0em">​</mo><mi>s</mi><mo lspace="0em" rspace="0em">​</mo><mi>d</mi></mrow></mrow></semantics></math> bytes<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>188 MB (70B, TP=8)PP bubble fraction<math alttext="(P-1)/(P+M-1)" display="inline"><semantics><mrow><mrow><mo stretchy="false">(</mo><mrow><mi>P</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>P</mi><mo>+</mo><mi>M</mi></mrow><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></semantics></math><math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math>=stages, <math alttext="M" display="inline"><semantics><mi>M</mi></semantics></math>=micro-batchesMFUobserved_toks <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> 6<math alttext="P" display="inline"><semantics><mi>P</mi></semantics></math> / peak_FLOPSTarget: <math alttext="&gt;40\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>40</mn><mo>%</mo></mrow></mrow></semantics></math>

## GPU Hardware Specs

GPUMemoryBW (HBM)BF16 TFLOPSNVLinkNotesA100-80GB80 GB HBM2e2.0 TB/s312600 GB/sWorkhorse, widely availableH100-80GB80 GB HBM33.35 TB/s989900 GB/sCurrent gen, FP8 supportH200-141GB141 GB HBM3e4.8 TB/s989900 GB/sLarge context / fewer GPUsB200192 GB HBM3e8.0 TB/s22501800 GB/sNext gen (2025)

## Hyperparameter Ranges

ParameterTypical RangeDefaultNotes<math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math> (DPO/KTO)0.05–0.50.1Higher = more conservative<math alttext="\epsilon" display="inline"><semantics><mi>ϵ</mi></semantics></math> (PPO clip)0.1–0.30.2Higher = more aggressive updates<math alttext="\gamma" display="inline"><semantics><mi>γ</mi></semantics></math> (GAE discount)0.99–1.01.0Use 1.0 for episodic tasks<math alttext="\lambda" display="inline"><semantics><mi>λ</mi></semantics></math> (GAE)0.9–0.990.95Lower = more biased, less varianceKL coeff (<math alttext="\beta_{\text{KL}}" display="inline"><semantics><msub><mi>β</mi><mtext>KL</mtext></msub></semantics></math>)0.01–0.20.05Auto-adapt to target KL <math alttext="\approx" display="inline"><semantics><mo>≈</mo></semantics></math> 5–8LR (RLHF)1e-7 – 5e-65e-7Much lower than pre-trainingLR (SFT)1e-5 – 5e-52e-5Standard fine-tuning rangeLoRA rank <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math>8–12816–64Higher <math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> = more capacity, more memoryLoRA alpha <math alttext="\alpha" display="inline"><semantics><mi>α</mi></semantics></math><math alttext="r" display="inline"><semantics><mi>r</mi></semantics></math> – <math alttext="2r" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>r</mi></mrow></semantics></math><math alttext="2r" display="inline"><semantics><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>r</mi></mrow></semantics></math>Scaling factor; <math alttext="\alpha/r" display="inline"><semantics><mrow><mi>α</mi><mo>/</mo><mi>r</mi></mrow></semantics></math> is the effective scaleTemperature (gen)0.6–1.00.7Lower = less diverse candidatesNum generations <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math>4–644–16For GRPO/Online DPO/Best-of-NGrad clip norm0.5–2.01.0Prevents gradient explosion

## TRL API Quick Reference

TrainerMethodKey ConfigData Format<code>SFTTrainer</code>Supervised FT<code>packing, max_seq_length</code>prompt + completion<code>RewardTrainer</code>Reward model<code>center_rewards_coefficient</code>prompt + chosen + rejected<code>PPOTrainer</code>PPO<code>init_kl_coef, target_kl, cliprange</code>prompts (online gen)<code>DPOTrainer</code>DPO/IPO<code>beta, loss_type="sigmoid"/"ipo"</code>prompt + chosen + rejected<code>GRPOTrainer</code>GRPO<code>num_generations, beta, use_vllm</code>prompts + reward_fn<code>OnlineDPOTrainer</code>Online DPO<code>num_generations, reward_model_path</code>prompts (online gen)<code>KTOTrainer</code>KTO<code>desirable_weight, undesirable_weight</code>prompt + completion + label<code>ORPOTrainer</code>ORPO<code>beta</code>prompt + chosen + rejected<code>Best-of-N (manual)</code>Best-of-N<code>n_samples</code>prompts (inference)

## RAG Pipeline Formulas



## Agentic Design Patterns

PatternStructureBest ForReActThink <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Act <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Observe <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> loopGeneral tool-use agentsPlan-and-ExecutePlan <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Execute steps <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ReviseLong-horizon, structured tasksSupervisorRouter <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> specialist agentsMulti-domain, clear subtask boundariesSwarm (handoffs)Agent transfers control + contextCustomer service, escalation flowsHierarchicalTree of delegating agentsComplex decompositionHuman-in-the-loopAgent <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> Approval gate <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ContinueHigh-stakes, irreversible actions

## Agent Communication Protocols

ProtocolScopeTransportKey ConceptMCPTool integrationstdio / HTTP+SSEServer exposes tools; client discovers &amp; callsA2AAgent-to-agentHTTP + JSON-RPCTasks with lifecycle (submitted<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>working<math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>done)OpenAI Function CallingTool useAPI payloadJSON schema in <code>tools[]</code> array

## Context Window Budget

<div class="hh-equation" id="ch29.e22"><math alttext="C\geq\underbrace{S}_{\text{system}}+\underbrace{M}_{\text{memory/RAG}}+\underbrace{T}_{\text{tool defs}}+\underbrace{H}_{\text{history}}+\underbrace{R}_{\text{reserved output}}" display="block"><semantics><mrow><mi>C</mi><mo>≥</mo><mrow><munder><munder accentunder="true"><mi>S</mi><mo stretchy="true">⏟</mo></munder><mtext>system</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>M</mi><mo stretchy="true">⏟</mo></munder><mtext>memory/RAG</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>T</mi><mo stretchy="true">⏟</mo></munder><mtext>tool defs</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>H</mi><mo stretchy="true">⏟</mo></munder><mtext>history</mtext></munder><mo>+</mo><munder><munder accentunder="true"><mi>R</mi><mo stretchy="true">⏟</mo></munder><mtext>reserved output</mtext></munder></mrow></mrow></semantics></math><span class="hh-equation-number">(29.22)</span></div>

Rule of thumb for 128K context:

<ul><li>System prompt: 1–4K tokens (fixed)</li><li>Tool definitions: 2–8K (scales with # tools)</li><li>RAG context: 4–16K (top-<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> chunks)</li><li>History: grows unbounded <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> summarize/truncate</li><li>Reserved output: 2–8K</li></ul>

## Common Failure Modes & Fixes

SymptomLikely CauseFixReward up, quality downReward hackingRM ensemble, length penalty, increase <math alttext="\beta" display="inline"><semantics><mi>β</mi></semantics></math>KL exploding (<math alttext="&gt;" display="inline"><semantics><mo>&gt;</mo></semantics></math>15)LR too high or mode collapseReduce LR, checkpoint rollbackEntropy collapsePremature convergenceIncrease entropy coeff, raise temperatureTraining loss NaNGradient explosionReduce LR, increase grad clip, check dataNo improvement after 5K stepsBad prompt distributionGoldilocks filter (20–80% pass rate)Benchmark regressionAlignment taxReduce RL budget, use LoRA, mix SFT dataLength increasing monotonicallyLength exploit in RMLength penalty, retrain RM with length controlOOM during generationKV cache overflowReduce batch, increase TP, PagedAttentionAgent loops foreverNo max-iteration guardSet <code>max_iterations</code>, add loop detectionTool call parse failuresInconsistent output formatFew-shot examples, constrained decodingRAG returns irrelevant docsPoor embedding / chunkingReranker, hybrid search, smaller chunksMulti-agent deadlockCircular dependenciesDAG enforcement, timeout per agent

## Method Selection Decision Tree

<ol><li>Have paired preferences (chosen + rejected)?<span class="hh-tag">•</span>Noisy labels <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>IPO<span class="hh-tag">•</span>Memory-constrained, no SFT done yet <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>ORPO<span class="hh-tag">•</span>Clean data, limited compute <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>DPO<span class="hh-tag">•</span>DPO plateaus, want exploration <math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>Online DPO</li><li>Have only binary feedback (thumbs up/down)?<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>KTO</li><li>Have verifiable rewards (math/code)?<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>GRPO</li><li>Need maximum quality, any cost?<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>PPO</li><li>Want training-free improvement?<math alttext="\rightarrow" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math>Best-of-N</li></ol>

## Evaluation Metrics

MetricRangeWhat It MeasuresPerplexity<math alttext="[1,\infty)" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>1</mn><mo>,</mo><mi mathvariant="normal">∞</mi><mo stretchy="false">)</mo></mrow></semantics></math>Model’s surprise; lower = better language modelingWin Rate (vs. baseline)<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Fraction of outputs preferred by judge/humanBLEU<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math><math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram overlap with reference (precision-focused)ROUGE-L<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Longest common subsequence with referencePass@<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math><math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Probability <math alttext="\geq" display="inline"><semantics><mo>≥</mo></semantics></math>1 of <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> code samples passes testsMMLU / GPQA<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Multi-choice accuracy on knowledge/reasoning benchmarksHumanEval<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Functional correctness of generated codeFaithfulness (RAG)<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Fraction of claims supported by retrieved contextContext Relevancy<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Fraction of retrieved content relevant to queryAnswer Relevancy<math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>Degree to which answer addresses the question

## Reasoning & Test-Time Scaling

MethodCompute CostMechanismChain-of-Thought (CoT)1.5–3<math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> tokens“Think step by step” in promptSelf-Consistency<math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> generationSample <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> CoT paths, majority vote on final answerTree-of-Thought (ToT)<math alttext="B\times D\times" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em" rspace="0.222em">×</mo><mi>D</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> generationBFS/DFS over reasoning tree; evaluate branchesBest-of-<math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math><math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> generationSample <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math>, score with RM, pick highestBeam search (on reasoning)<math alttext="B\times" display="inline"><semantics><mrow><mi>B</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> generationMaintain top-<math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> partial reasoning chainsBudget forcingVariableAllocate more tokens to harder problems dynamicallyVerification (ORM/PRM)<math alttext="N\times" display="inline"><semantics><mrow><mi>N</mi><mo lspace="0.222em">×</mo></mrow></semantics></math> gen + scoringGenerate <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> solutions, rank by outcome/process RM

## Memory System Types

TypeStorageUse CaseWorking memoryContext windowCurrent conversation, immediate tool resultsEpisodic memoryVector storePast interactions, user preferences, session historySemantic memoryKnowledge graph / embeddingsFacts, concepts, domain knowledgeProcedural memorySkill library / codeHow-to procedures, learned workflows

## MCP Quick Reference

PrimitiveDirectionSide Effects?PurposeToolsClient <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ServerYesExecute actions (create, modify, delete)ResourcesClient <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ServerNo (read-only)Read data (files, DB records, configs)PromptsClient <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ServerNoReusable templates for common tasksSamplingServer <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> ClientNoServer requests LLM generation from client

Transport: <code>stdio</code> (local subprocess) or <code>HTTP+SSE</code> (remote, streamable). <br>

Discovery: Client calls <code>tools/list</code>, <code>resources/list</code>, <code>prompts/list</code> at connection init. <br>

Tool annotations: <code>readOnlyHint</code>, <code>destructiveHint</code>, <code>idempotentHint</code>, <code>openWorldHint</code>.

## A2A Protocol Quick Reference

ConceptDescriptionAgent CardJSON at <code>/.well-known/agent.json</code> — name, skills, supported content typesTaskUnit of work: <code>id</code>, <code>status</code>, <code>artifacts</code>. Lifecycle: submitted <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> working <math alttext="\to" display="inline"><semantics><mo stretchy="false">→</mo></semantics></math> completed/failedMessageCommunication unit within a task (role: user/agent, parts: text/file/data)ArtifactOutput produced by the agent (structured data, files, generated content)Push NotificationsWebhook-based updates for long-running tasks (via <code>tasks/pushNotification/set</code>)

Key endpoints: <code>tasks/send</code> (create/update), <code>tasks/get</code> (poll status), <code>tasks/sendSubscribe</code> (SSE stream).

## Agent Framework Comparison

FrameworkOrchestrationMulti-AgentBest ForLangGraphExplicit state graphConditional routingProduction: persistence, HITL, fine controlOpenAI Agents SDKDeclarative handoffsHandoff-basedSimplicity: guardrails, tracing, fast startAutoGen (AG2)Conversation-drivenGroupChatPrototyping: code execution, researchCrewAIRole-based teamsSequential/parallelLow-code: quick demos, simple pipelinesGoogle ADKSession + eventsA2A nativeEnterprise: artifact mgmt, multi-modal

## Agentic RL Formulas



## Agent Security Checklist

ThreatLayerMitigationPrompt injection (direct)InputInput validation, instruction hierarchy, delimitersPrompt injection (indirect)Tool outputTreat tool output as untrusted; don’t follow instructions in retrieved docsTool misuseExecutionLeast-privilege permissions; <code>destructiveHint</code> gates; sandboxingData exfiltrationOutputOutput filtering; restrict tool access to allowed domainsExcessive autonomyArchitectureMax iterations; cost budgets; human approval gatesConfused deputyMulti-agentVerify task origin; capability-based access control

## Agent Evaluation Metrics

MetricFormula / DefinitionTargetTask Success Rate (TSR)Correct completions / total tasks<math alttext="&gt;85\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>85</mn><mo>%</mo></mrow></mrow></semantics></math> (production)Steps to completionAvg agent actions per successful taskLower = more efficientCost per taskTotal tokens <math alttext="\times" display="inline"><semantics><mo>×</mo></semantics></math> price/tokenBudget-dependentLatency (TTFC)Time from request to first useful output<math alttext="&lt;5" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mn>5</mn></mrow></semantics></math>s for interactiveTool call accuracyCorrect tool selections / total calls<math alttext="&gt;90\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>90</mn><mo>%</mo></mrow></mrow></semantics></math>Recovery rateSuccessful retries / initial failures<math alttext="&gt;60\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>60</mn><mo>%</mo></mrow></mrow></semantics></math>Human escalation rateTasks requiring human / total tasks<math alttext="&lt;15\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&lt;</mo><mrow><mn>15</mn><mo>%</mo></mrow></mrow></semantics></math>

## Key Agentic Benchmarks

BenchmarkDomainMetricSOTA (2025)SWE-bench VerifiedSoftware engineering% resolved issues<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>70%WebArenaWeb browsingTask success rate<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>40%OSWorldDesktop computer useTask success rate<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>25%GAIAGeneral AI assistantExact match accuracy<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>75% (L1)Tau-benchTool-use reliabilityPass rate (5 trials)<math alttext="\sim" display="inline"><semantics><mo>∼</mo></semantics></math>65%HumanEval / MBPPCode generationPass@1<math alttext="&gt;95\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>95</mn><mo>%</mo></mrow></mrow></semantics></math>

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">The Hitchhiker's Guide to Agentic AI · Chapter 30 Conclusion and Future Directions</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>

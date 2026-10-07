---
title: "第 14 章 LLM 评估"
slug: "hitchhiker-agentic-ai-14-llm-evaluation"
lang: "zh"
date: "2026-09-16T00:15:00.000Z"
draft: false
featured: false
categories:
  - "hitchhiker-agentic-ai"
cover: "./cover.png"
description: "评估是任何严谨机器学习流水线的支柱，但它或许是大语言模型开发中最被低估的组成部分。与经典监督学习不同——在那里，带有真实标签的留出测试集能提供干净的信号——评估 LLM 需要应对开放式生成、主观质量判断、多步推理链，以及无处不在的基准污染风险。…"
keywords:
  - "智能体"
  - "Agentic AI"
  - "LLM"
  - "大语言模型"
  - "强化学习"
  - "RLHF"
  - "MCP"
---

<div class="hh-guide">

<aside class="hh-part">
<p class="hh-part-title">第 IV 部分 评估</p>
</aside>

评估是任何严谨机器学习流水线的支柱，但它或许是大语言模型开发中最被低估的组成部分。与经典监督学习不同——在那里，带有真实标签的留出测试集能提供干净的信号——评估 LLM 需要应对开放式生成、主观质量判断、多步推理链，以及无处不在的基准污染风险。本节系统性地梳理评估领域的全景：从评估类型的分类法、人工标注的机制，到排序指标的数学原理与 LLM-as-judge 的实操要点，再到那些悄无声息地腐蚀评估流水线的陷阱。

## 评估方案设计

在收集任何一个数据点之前，从业者必须先决定要测量<em>什么</em>以及<em>如何</em>测量。一套有原则的分类法能避免一个常见错误：仅凭便利性而非与部署目标的对齐性来选择指标。

### 评估类型分类法

<em>内在（Intrinsic）</em>评估孤立地度量模型输出的性质，不参照下游应用。留出语料上的困惑度、相对参考译文的 BLEU 分数、编码基准上的 pass@<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math>，都属于内在评估。<em>外在（Extrinsic）</em>评估则度量模型对真实任务或系统的影响：将 LLM 接入客服流水线后，工单升级率是否下降？编码助手是否提升了开发者的开发速度？

<em>自动</em>评估使用确定性函数（BLEU、精确匹配）或学习得到的模型（BERTScore、LLM-as-judge），无需人工参与即可对输出打分。<em>人工</em>评估则由标注者对模型输出进行打分或排序。表 表 14.1 总结了二者的权衡。

<div class="hh-table" id="ch14.t1"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>类型</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>成本</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>速度</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>可复现性</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>效度</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">自动（基于规则）</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极快</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>完美</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低–中</span></span></td></tr><tr><th class="ltx_align_left">自动（基于模型）</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>快</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中–高</span></span></td></tr><tr><th class="ltx_align_left">众包人工</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>数天</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td></tr><tr><th class="ltx_align_left">专家人工</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>数周</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低–中</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td></tr><tr><th class="ltx_align_left">外在 / A/B 测试</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>数月</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极高</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 14.1：</span>评估方法分类法及其关键权衡。</p></div>

基于参考的指标（BLEU、ROUGE、BERTScore）将模型输出与一个或多个金标准参考进行比较。无参考指标（困惑度、LLM-as-judge、人类偏好）则无需参考即可评估质量。当输出空间过大、无法穷尽收集参考时（例如开放式对话），无参考方法是必不可少的。

### 何时使用何种方法

一个实用的决策框架：

<ul><li>若任务有明确正确答案（数学、代码、事实型问答）：使用精确匹配或基于执行的指标。</li><li>若任务是开放式的但存在参考输出：使用基于参考的指标作为下界，并辅以 LLM-as-judge。</li><li>若任务是主观的（有用性、语气、创造性）：使用人工评估或经过良好校准的 LLM 评判者。</li><li>若任务涉及多步 Agent 行为：使用任务成功率与轨迹效率（第 14.6 Agent 任务的指标 节）。</li></ul>

## 评估数据采集

高质量的评估数据是可信基准的基础。本节涵盖人工标注流水线的设计、标注质量的统计度量，以及在众包与专家标注之间如何取舍。

### 人工标注流水线

一条健壮的标注流水线包含五个阶段：

<ol><li>任务定义。精确说明标注任务：评分对象是什么、采用何种量表、依据哪些标准。此阶段的歧义会传导为带噪标签。</li><li>规范编写。撰写标注规范，并配以覆盖边界情形的详细示例。在全面部署前先用小规模试点组反复打磨。</li><li>标注者招募与培训。选择具备相应背景知识的标注者。组织一次校准会议，让标注者标注相同样本并讨论分歧。</li><li>质量控制。在标注队列中嵌入带已知标签的金标样本。对在金标样本上准确率低于阈值的标注者进行标记。</li><li>聚合。使用多数投票、平均，或概率模型（如 Dawid–Skene）来合并每条样本的多次标注。</li></ol>

### 标注者间一致性

原始一致率（所有标注者达成一致的样本比例）是一个不充分的度量，因为它没有考虑偶然一致。两个标准的偶然性修正度量是 Cohen’s <math alttext="\kappa" display="inline"><semantics><mi>κ</mi></semantics></math>[60]（两位标注者）和 Fleiss’ <math alttext="\kappa" display="inline"><semantics><mi>κ</mi></semantics></math>[92]（多位标注者）。

给定两位标注者将 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个样本标注为 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个类别，令 <math alttext="p_{o}" display="inline"><semantics><msub><mi>p</mi><mi>o</mi></msub></semantics></math> 为观察到的一致率，<math alttext="p_{e}" display="inline"><semantics><msub><mi>p</mi><mi>e</mi></msub></semantics></math> 为独立性假设下的期望一致率：

<div class="hh-equation" id="ch14.e1"><math alttext="\kappa=\frac{p_{o}-p_{e}}{1-p_{e}}" display="block"><semantics><mrow><mi>κ</mi><mo>=</mo><mfrac><mrow><msub><mi>p</mi><mi>o</mi></msub><mo>−</mo><msub><mi>p</mi><mi>e</mi></msub></mrow><mrow><mn>1</mn><mo>−</mo><msub><mi>p</mi><mi>e</mi></msub></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.1)</span></div>

其中

<div class="hh-equation" id="ch14.e2"><math alttext="p_{o}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{annotator 1 agrees with annotator 2 on item }i]" display="block"><semantics><mrow><msub><mi>p</mi><mi>o</mi></msub><mo>=</mo><mrow><mfrac><mn>1</mn><mi>N</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mrow><mn>𝟏</mn><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><mtext>annotator 1 agrees with annotator 2 on item </mtext><mo lspace="0em" rspace="0em">​</mo><mi>i</mi></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.2)</span></div>

以及

<div class="hh-equation" id="ch14.e3"><math alttext="p_{e}=\sum_{c=1}^{k}p_{1c}\cdot p_{2c}" display="block"><semantics><mrow><msub><mi>p</mi><mi>e</mi></msub><mo rspace="0.111em">=</mo><mrow><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>c</mi><mo>=</mo><mn>1</mn></mrow><mi>k</mi></munderover><msub><mi>p</mi><mrow><mn>1</mn><mo lspace="0em" rspace="0em">​</mo><mi>c</mi></mrow></msub></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><msub><mi>p</mi><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>c</mi></mrow></msub></mrow></mrow></semantics></math><span class="hh-equation-number">(14.3)</span></div>

其中 <math alttext="p_{jc}" display="inline"><semantics><msub><mi>p</mi><mrow><mi>j</mi><mo lspace="0em" rspace="0em">​</mo><mi>c</mi></mrow></msub></semantics></math> 是标注者 <math alttext="j" display="inline"><semantics><mi>j</mi></semantics></math> 将样本判为类别 <math alttext="c" display="inline"><semantics><mi>c</mi></semantics></math> 的比例。Cohen’s <math alttext="\kappa" display="inline"><semantics><mi>κ</mi></semantics></math> 的取值从 <math alttext="-1" display="inline"><semantics><mrow><mo>−</mo><mn>1</mn></mrow></semantics></math>（完全不一致）经 <math alttext="0" display="inline"><semantics><mn>0</mn></semantics></math>（偶然一致）到 <math alttext="1" display="inline"><semantics><mn>1</mn></semantics></math>（完全一致）。一般认为 <math alttext="0.6" display="inline"><semantics><mn>0.6</mn></semantics></math> 以上可以接受；<math alttext="0.8" display="inline"><semantics><mn>0.8</mn></semantics></math> 以上为强一致。

对于 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 位标注者将 <math alttext="N" display="inline"><semantics><mi>N</mi></semantics></math> 个样本标注为 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个类别，令 <math alttext="n_{ij}" display="inline"><semantics><msub><mi>n</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub></semantics></math> 为将样本 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 标注为类别 <math alttext="j" display="inline"><semantics><mi>j</mi></semantics></math> 的标注者人数。定义：

<div class="hh-equation" id="ch14.e4"><math alttext="\bar{P}_{i}=\frac{1}{n(n-1)}\sum_{j=1}^{k}n_{ij}(n_{ij}-1),\qquad\bar{P}=\frac{1}{N}\sum_{i=1}^{N}\bar{P}_{i}" display="block"><semantics><mrow><mrow><msub><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mi>i</mi></msub><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mi>n</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>n</mi><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>k</mi></munderover><mrow><msub><mi>n</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>n</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub><mo>−</mo><mn>1</mn></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><mo rspace="2.167em">,</mo><mrow><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mo>=</mo><mrow><mfrac><mn>1</mn><mi>N</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msub><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mi>i</mi></msub></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.4)</span></div>

<div class="hh-equation" id="ch14.e5"><math alttext="\bar{P}_{j}^{e}=\frac{1}{Nn}\sum_{i=1}^{N}n_{ij},\qquad P_{e}=\sum_{j=1}^{k}\left(\bar{P}_{j}^{e}\right)^{2}" display="block"><semantics><mrow><mrow><msubsup><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mi>j</mi><mi>e</mi></msubsup><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mi>N</mi><mo lspace="0em" rspace="0em">​</mo><mi>n</mi></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><msub><mi>n</mi><mrow><mi>i</mi><mo lspace="0em" rspace="0em">​</mo><mi>j</mi></mrow></msub></mrow></mrow></mrow><mo rspace="2.167em">,</mo><mrow><msub><mi>P</mi><mi>e</mi></msub><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>k</mi></munderover><msup><mrow><mo>(</mo><msubsup><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mi>j</mi><mi>e</mi></msubsup><mo>)</mo></mrow><mn>2</mn></msup></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.5)</span></div>

<div class="hh-equation" id="ch14.e6"><math alttext="\kappa_{F}=\frac{\bar{P}-P_{e}}{1-P_{e}}" display="block"><semantics><mrow><msub><mi>κ</mi><mi>F</mi></msub><mo>=</mo><mfrac><mrow><mover accent="true"><mi>P</mi><mo>¯</mo></mover><mo>−</mo><msub><mi>P</mi><mi>e</mi></msub></mrow><mrow><mn>1</mn><mo>−</mo><msub><mi>P</mi><mi>e</mi></msub></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.6)</span></div>

### 标注规范设计

有效的标注规范共享若干特征：

<ul><li>可操作的标准。用具体、可观察的行为替换诸如“有帮助”之类的模糊措辞：“回答直接回应用户的问题，并提供完成所述任务所需的全部信息。”</li><li>详细示例。每个评分等级至少提供两个示例，其中包括边界情形。</li><li>决策树。对于复杂任务，用一张引导标注者依次作出二元判断的流程图可降低认知负荷并提升一致性。</li><li>明确范围。说明标注者<em>不应</em>考虑的因素（例如：“不要因风格偏好而扣分；只关注事实准确性”）。</li></ul>

### 众包标注与专家标注

<div class="hh-table" id="ch14.t2"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>维度</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>众包</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>专家标注</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">每条样本成本</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低（<math alttext="0.01--" display="inline"><semantics><mrow><mrow><mn>0.01</mn><mo rspace="0em">−</mo></mrow><mo lspace="0em">−</mo></mrow><span></span></semantics></math>0.10）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高（<math alttext="1--" display="inline"><semantics><mrow><mrow><mn>1</mn><mo rspace="0em">−</mo></mrow><mo lspace="0em">−</mo></mrow><span></span></semantics></math>50）</span></span></td></tr><tr><th class="ltx_align_left">吞吐量</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极高</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td></tr><tr><th class="ltx_align_left">领域知识</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td></tr><tr><th class="ltx_align_left">一致性</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>可变</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td></tr><tr><th class="ltx_align_left">适用任务</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>简单偏好、流畅度</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>技术准确性、安全性</span></span></td></tr><tr><th class="ltx_align_left">平台</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>MTurk、Prolific、Scale AI</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>领域专家、内部团队</span></span></td></tr><tr><th class="ltx_align_left">质量控制</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>金标样本、注意力检查</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>校准会议、同行评审</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 14.2：</span>LLM 评估中众包标注与专家标注的比较。</p></div>

对于安全关键型评估（例如检测有害输出、评估医疗建议），专家标注不可替代。对于大规模偏好收集（例如构建奖励模型训练集），带严格质量控制的众包往往是唯一可行的选择。

## 评估用合成数据生成

人工标注昂贵且缓慢。合成数据生成利用 LLM 自身大规模产出评估数据。本节涵盖主要的范式。

### 用于校准的 LLM-as-Judge

使用 LLM 生成评估标签时，校准至关重要：评判者的分数必须与人类判断对齐。令 <math alttext="h_{i}\in[0,1]" display="inline"><semantics><mrow><msub><mi>h</mi><mi>i</mi></msub><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> 为样本 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 的人类偏好分数，<math alttext="\hat{h}_{i}" display="inline"><semantics><msub><mover accent="true"><mi>h</mi><mo>^</mo></mover><mi>i</mi></msub></semantics></math> 为评判者的预测分数。校准误差由期望校准误差（ECE）[124] 度量：

<div class="hh-equation" id="ch14.e7"><math alttext="\text{ECE}=\sum_{b=1}^{B}\frac{|B_{b}|}{n}\left|\text{acc}(B_{b})-\text{conf}(B_{b})\right|" display="block"><semantics><mrow><mtext>ECE</mtext><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>b</mi><mo>=</mo><mn>1</mn></mrow><mi>B</mi></munderover><mrow><mfrac><mrow><mo stretchy="false">|</mo><msub><mi>B</mi><mi>b</mi></msub><mo stretchy="false">|</mo></mrow><mi>n</mi></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><mo>|</mo><mrow><mrow><mtext>acc</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>B</mi><mi>b</mi></msub><mo stretchy="false">)</mo></mrow></mrow><mo>−</mo><mrow><mtext>conf</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>B</mi><mi>b</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>|</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.7)</span></div>

其中 <math alttext="B_{b}" display="inline"><semantics><msub><mi>B</mi><mi>b</mi></msub></semantics></math> 是第 <math alttext="b" display="inline"><semantics><mi>b</mi></semantics></math> 个置信度分箱，<math alttext="\text{acc}(B_{b})" display="inline"><semantics><mrow><mtext>acc</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>B</mi><mi>b</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是该分箱中评判者与人类一致样本所占的比例，<math alttext="\text{conf}(B_{b})" display="inline"><semantics><mrow><mtext>conf</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>B</mi><mi>b</mi></msub><mo stretchy="false">)</mo></mrow></mrow></semantics></math> 是该分箱中评判者置信度的均值。

一个校准良好的评判者对任意 <math alttext="p\in[0,1]" display="inline"><semantics><mrow><mi>p</mi><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math> 都满足 <math alttext="\mathbb{E}[\hat{h}_{i}\mid\hat{h}_{i}=p]=p" display="inline"><semantics><mrow><mrow><mi>𝔼</mi><mo>⁡</mo><mrow><mo stretchy="false">[</mo><mrow><msub><mover accent="true"><mi>h</mi><mo>^</mo></mover><mi>i</mi></msub><mo lspace="0em" rspace="0.167em">∣</mo><mrow><msub><mover accent="true"><mi>h</mi><mo>^</mo></mover><mi>i</mi></msub><mo>=</mo><mi>p</mi></mrow></mrow><mo stretchy="false">]</mo></mrow></mrow><mo>=</mo><mi>p</mi></mrow></semantics></math>。校准可以通过温度缩放改善：将评判者的原始 logit <math alttext="z" display="inline"><semantics><mi>z</mi></semantics></math> 替换为 <math alttext="z/T" display="inline"><semantics><mrow><mi>z</mi><mo>/</mo><mi>T</mi></mrow></semantics></math>，其中 <math alttext="T" display="inline"><semantics><mi>T</mi></semantics></math> 在留出校准集上调优，以最小化负对数似然。

### Self-Instruct

Self-Instruct [371] 从一小批人工编写的种子任务中自举出指令遵循数据。算法如下：

<ol><li>维护一个初始包含 <math alttext="175" display="inline"><semantics><mn>175</mn></semantics></math> 个种子任务的任务池。</li><li>从任务池中采样 <math alttext="8" display="inline"><semantics><mn>8</mn></semantics></math> 个任务；将它们作为 few-shot 示例提示 LLM 生成新任务。</li><li>过滤生成的任务：去除近似重复项（与任何已有任务的 ROUGE-L 相似度 <math alttext="&gt;0.7" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mn>0.7</mn></mrow></semantics></math>），将其分类为分类任务或非分类任务，并生成输入—输出实例。</li><li>将接受的任务加入任务池。</li><li>重复直至达到期望的任务池规模。</li></ol>

### Evol-Instruct

Evol-Instruct [398] 通过迭代改写指令使其更复杂或更多样，从而演化种子指令集。它应用两种演化算子：

<ul><li>深度演化：增加约束、增加推理步数、把抽象具体化、加深领域知识要求。</li><li>广度演化：围绕相关但不同的主题生成新指令，提升主题多样性。</li></ul>

指令只有在通过淘汰过滤器后才会被接受：演化后的指令不能是简单的复制，不能包含“I’m sorry”或类似的拒绝措辞，且不能比原指令更短。

### Constitutional AI 数据生成

Constitutional AI（CAI）[16] 通过让模型依据一组原则（即“宪法”）批判并修订自身输出，来生成偏好数据。流水线如下：

<ol><li>监督学习阶段：采样一个有害 Prompt，生成初始回答，然后提示模型依据一条宪法原则批判该回答并加以修订。将修订后的回答用作监督微调的目标。</li><li>RL 阶段：生成成对的回答（原始 vs. 修订后），用模型标注哪一条更符合宪法，并在这些标注上训练偏好模型。将该偏好模型用作 RLHF 的奖励信号。</li></ol>

该方法无需人工标注有害内容即可生成偏好数据，减少了标注者接触令人痛苦材料的风险。

### 评估数据的蒸馏

一个强大的教师模型（如 GPT-4）可以为训练更小的评判模型生成高质量评估数据。蒸馏流水线如下：

<ol><li>收集一组多样化的 Prompt 与模型回答。</li><li>用教师生成详尽的判断（分数 + 理由）。</li><li>在（prompt, response, judgment）三元组上微调一个更小的模型。</li><li>用留出的人工标注验证学生评判者。</li></ol>

### Arena 风格的成对生成

Chatbot Arena [433] 通过一个众包对战平台生成评估数据：用户提交 Prompt，并对两个匿名模型回答中自己更偏好哪一个进行投票。由此产生大规模、自然多样化的成对偏好数据集。关键设计选择包括：

<ul><li>匿名化：隐藏模型身份以防止品牌偏差。</li><li>用户提交的 Prompt：确保 Prompt 的多样性与真实世界相关性。</li><li>平局处理：用户可以判定平局，或表示两个回答都不好。</li><li>去重：过滤近似重复的 Prompt，以防常见查询被过度代表。</li></ul>

## 排序任务的指标

当目标是对模型按质量排序时，成对比较数据比绝对分数更可靠。本节推导 LLM 评估中使用的主要排序系统。

### ELO 评分系统

ELO 系统 [84] 最初为国际象棋开发，它为每位选手（模型）赋予一个标量评分 <math alttext="R" display="inline"><semantics><mi>R</mi></semantics></math>，使得选手 <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> 对选手 <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 的期望得分为：

<div class="hh-equation" id="ch14.e8"><math alttext="E_{A}=\frac{1}{1+10^{(R_{B}-R_{A})/400}}" display="block"><semantics><mrow><msub><mi>E</mi><mi>A</mi></msub><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><msup><mn>10</mn><mrow><mrow><mo stretchy="false">(</mo><mrow><msub><mi>R</mi><mi>B</mi></msub><mo>−</mo><msub><mi>R</mi><mi>A</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo>/</mo><mn>400</mn></mrow></msup></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.8)</span></div>

ELO 模型假设每位选手在某局比赛中的表现是一个服从逻辑斯谛分布（logistic distribution）、中心位于其评分的随机变量。<math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> 击败 <math alttext="B" display="inline"><semantics><mi>B</mi></semantics></math> 的概率为：

<div class="hh-equation" id="ch14.e9"><math alttext="P(A\succ B)=\sigma\!\left(\frac{R_{A}-R_{B}}{s}\right)=\frac{1}{1+e^{-(R_{A}-R_{B})/s}}" display="block"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>A</mi><mo>≻</mo><mi>B</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mpadded width="0.437em"><mi>σ</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>R</mi><mi>A</mi></msub><mo>−</mo><msub><mi>R</mi><mi>B</mi></msub></mrow><mi>s</mi></mfrac><mo>)</mo></mrow></mrow><mo>=</mo><mfrac><mn>1</mn><mrow><mn>1</mn><mo>+</mo><mi>e</mi><msup><mrow></mrow><mrow><mo>−</mo><mo stretchy="false">(</mo><mi>R</mi><msub><mrow></mrow><mi>A</mi></msub><mo>−</mo><mi>R</mi><msub><mrow></mrow><mi>B</mi></msub><mo stretchy="false">)</mo><mo>/</mo><mi>s</mi></mrow></msup></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.9)</span></div>

其中 <math alttext="s=400/\ln(10)\approx 173.7" display="inline"><semantics><mrow><mi>s</mi><mo>=</mo><mrow><mn>400</mn><mo>/</mo><mrow><mi>ln</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>10</mn><mo stretchy="false">)</mo></mrow></mrow></mrow><mo>≈</mo><mn>173.7</mn></mrow></semantics></math> 是尺度参数，其取值使得 400 分的分差对应 <math alttext="10:1" display="inline"><semantics><mrow><mn>10</mn><mo lspace="0.278em" rspace="0.278em">:</mo><mn>1</mn></mrow></semantics></math> 的胜算比（odds ratio）。每局比赛结束后，依据结果 <math alttext="S_{A}\in\{0,0.5,1\}" display="inline"><semantics><mrow><msub><mi>S</mi><mi>A</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mrow><mn>0</mn><mo>,</mo><mn>0.5</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math>（负、平、胜）更新评分：

<div class="hh-equation" id="ch14.e10"><math alttext="R_{A}\leftarrow R_{A}+K(S_{A}-E_{A}),\qquad R_{B}\leftarrow R_{B}+K(S_{B}-E_{B})" display="block"><semantics><mrow><mrow><msub><mi>R</mi><mi>A</mi></msub><mo stretchy="false">←</mo><mrow><msub><mi>R</mi><mi>A</mi></msub><mo>+</mo><mrow><mi>K</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>S</mi><mi>A</mi></msub><mo>−</mo><msub><mi>E</mi><mi>A</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo rspace="2.167em">,</mo><mrow><msub><mi>R</mi><mi>B</mi></msub><mo stretchy="false">←</mo><mrow><msub><mi>R</mi><mi>B</mi></msub><mo>+</mo><mrow><mi>K</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>S</mi><mi>B</mi></msub><mo>−</mo><msub><mi>E</mi><mi>B</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.10)</span></div>

其中 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 是控制学习率的 <math alttext="K" display="inline"><semantics><mi>K</mi></semantics></math> 因子。在 Chatbot Arena 中，<math alttext="K=4" display="inline"><semantics><mrow><mi>K</mi><mo>=</mo><mn>4</mn></mrow></semantics></math> 被采用。

由于 ELO 评分依赖于比赛的处理顺序，置信区间通过自举重采样计算：对对战日志进行有放回重采样 <math alttext="B=1000" display="inline"><semantics><mrow><mi>B</mi><mo>=</mo><mn>1000</mn></mrow></semantics></math> 次，对每个重采样从头重新计算 ELO 评分，并报告第 2.5 与第 97.5 百分位数作为 95% 置信区间。

### Bradley–Terry 模型

Bradley–Terry（BT）模型 [29] 是 ELO 的极大似然替代方案。给定 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 个强度参数为 <math alttext="\beta_{1},\ldots,\beta_{n}&gt;0" display="inline"><semantics><mrow><msub><mi>β</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><msub><mi>β</mi><mi>n</mi></msub><mo>&gt;</mo><mn>0</mn></mrow></mrow></semantics></math> 的模型，模型 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 击败模型 <math alttext="j" display="inline"><semantics><mi>j</mi></semantics></math> 的概率为：

<div class="hh-equation" id="ch14.e11"><math alttext="P(i\succ j)=\frac{\beta_{i}}{\beta_{i}+\beta_{j}}" display="block"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>i</mi><mo>≻</mo><mi>j</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mfrac><msub><mi>β</mi><mi>i</mi></msub><mrow><msub><mi>β</mi><mi>i</mi></msub><mo>+</mo><msub><mi>β</mi><mi>j</mi></msub></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.11)</span></div>

给定一组成对结果 <math alttext="\{(i_{k},j_{k},y_{k})\}_{k=1}^{M}" display="inline"><semantics><msubsup><mrow><mo stretchy="false">{</mo><mrow><mo stretchy="false">(</mo><msub><mi>i</mi><mi>k</mi></msub><mo>,</mo><msub><mi>j</mi><mi>k</mi></msub><mo>,</mo><msub><mi>y</mi><mi>k</mi></msub><mo stretchy="false">)</mo></mrow><mo stretchy="false">}</mo></mrow><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>M</mi></msubsup></semantics></math>，其中当 <math alttext="i_{k}" display="inline"><semantics><msub><mi>i</mi><mi>k</mi></msub></semantics></math> 击败 <math alttext="j_{k}" display="inline"><semantics><msub><mi>j</mi><mi>k</mi></msub></semantics></math> 时 <math alttext="y_{k}=1" display="inline"><semantics><mrow><msub><mi>y</mi><mi>k</mi></msub><mo>=</mo><mn>1</mn></mrow></semantics></math>，否则为 <math alttext="y_{k}=0" display="inline"><semantics><mrow><msub><mi>y</mi><mi>k</mi></msub><mo>=</mo><mn>0</mn></mrow></semantics></math>，对数似然为：

<div class="hh-equation" id="ch14.e12"><math alttext="\ell(\bm{\beta})=\sum_{k=1}^{M}\left[y_{k}\log\frac{\beta_{i_{k}}}{\beta_{i_{k}}+\beta_{j_{k}}}+(1-y_{k})\log\frac{\beta_{j_{k}}}{\beta_{i_{k}}+\beta_{j_{k}}}\right]" display="block"><semantics><mrow><mrow><mi mathvariant="normal">ℓ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>𝜷</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.111em">=</mo><mrow><munderover><mo movablelimits="false" rspace="0em">∑</mo><mrow><mi>k</mi><mo>=</mo><mn>1</mn></mrow><mi>M</mi></munderover><mrow><mo>[</mo><mrow><mrow><msub><mi>y</mi><mi>k</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><msub><mi>β</mi><msub><mi>i</mi><mi>k</mi></msub></msub><mrow><msub><mi>β</mi><msub><mi>i</mi><mi>k</mi></msub></msub><mo>+</mo><msub><mi>β</mi><msub><mi>j</mi><mi>k</mi></msub></msub></mrow></mfrac></mrow></mrow><mo>+</mo><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><msub><mi>y</mi><mi>k</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><mfrac><msub><mi>β</mi><msub><mi>j</mi><mi>k</mi></msub></msub><mrow><msub><mi>β</mi><msub><mi>i</mi><mi>k</mi></msub></msub><mo>+</mo><msub><mi>β</mi><msub><mi>j</mi><mi>k</mi></msub></msub></mrow></mfrac></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.12)</span></div>

MLE <math alttext="\hat{\bm{\beta}}" display="inline"><semantics><mover accent="true"><mi>𝜷</mi><mo>^</mo></mover></semantics></math> 通过迭代缩放或梯度上升求得。BT 模型仅在相差一个乘性常数意义下可辨识；常用的归一化是 <math alttext="\sum_{i}\log\beta_{i}=0" display="inline"><semantics><mrow><mrow><msub><mo>∑</mo><mi>i</mi></msub><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>β</mi><mi>i</mi></msub></mrow></mrow><mo>=</mo><mn>0</mn></mrow></semantics></math>。在 log 空间中以 <math alttext="\theta_{i}=\log\beta_{i}" display="inline"><semantics><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo>=</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>β</mi><mi>i</mi></msub></mrow></mrow></semantics></math> 表示可得：

<div class="hh-equation" id="ch14.e13"><math alttext="P(i\succ j)=\sigma(\theta_{i}-\theta_{j})" display="block"><semantics><mrow><mrow><mi>P</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><mi>i</mi><mo>≻</mo><mi>j</mi></mrow><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mi>σ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mrow><msub><mi>θ</mi><mi>i</mi></msub><mo>−</mo><msub><mi>θ</mi><mi>j</mi></msub></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.13)</span></div>

这等价于一个带样本特定截距的逻辑斯谛回归。当完整的对战历史可用时，BT 模型优于 ELO，因为它同时利用全部数据，而不是逐局顺序处理。

### TrueSkill

TrueSkill [138] 是一种贝叶斯技能评分系统，它将每位选手的技能建模为高斯随机变量 <math alttext="s_{i}\sim\mathcal{N}(\mu_{i},\sigma_{i}^{2})" display="inline"><semantics><mrow><msub><mi>s</mi><mi>i</mi></msub><mo>∼</mo><mrow><mi>𝒩</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mi>μ</mi><mi>i</mi></msub><mo>,</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math>。选手 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 在某局比赛中的表现为 <math alttext="p_{i}=s_{i}+\epsilon_{i}" display="inline"><semantics><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>=</mo><mrow><msub><mi>s</mi><mi>i</mi></msub><mo>+</mo><msub><mi>ϵ</mi><mi>i</mi></msub></mrow></mrow></semantics></math>，其中 <math alttext="\epsilon_{i}\sim\mathcal{N}(0,\beta^{2})" display="inline"><semantics><mrow><msub><mi>ϵ</mi><mi>i</mi></msub><mo>∼</mo><mrow><mi>𝒩</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><msup><mi>β</mi><mn>2</mn></msup><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 是比赛特定的噪声。若 <math alttext="p_{i}&gt;p_{j}" display="inline"><semantics><mrow><msub><mi>p</mi><mi>i</mi></msub><mo>&gt;</mo><msub><mi>p</mi><mi>j</mi></msub></mrow></semantics></math>，则选手 <math alttext="i" display="inline"><semantics><mi>i</mi></semantics></math> 击败选手 <math alttext="j" display="inline"><semantics><mi>j</mi></semantics></math>。

观测到 <math alttext="i\succ j" display="inline"><semantics><mrow><mi>i</mi><mo>≻</mo><mi>j</mi></mrow></semantics></math> 之后的后验更新通过期望传播（EP）计算。针对获胜者的关键更新方程为：

<div class="hh-equation" id="ch14.e14"><math alttext="\mu_{i}\leftarrow\mu_{i}+\frac{\sigma_{i}^{2}}{c}\cdot v\!\left(\frac{\mu_{i}-\mu_{j}}{c}\right)" display="block"><semantics><mrow><msub><mi>μ</mi><mi>i</mi></msub><mo stretchy="false">←</mo><mrow><msub><mi>μ</mi><mi>i</mi></msub><mo>+</mo><mrow><mfrac><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mi>c</mi></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mpadded width="0.351em"><mi>v</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>μ</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>j</mi></msub></mrow><mi>c</mi></mfrac><mo>)</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.14)</span></div>

<div class="hh-equation" id="ch14.e15"><math alttext="\sigma_{i}^{2}\leftarrow\sigma_{i}^{2}\left[1-\frac{\sigma_{i}^{2}}{c^{2}}\cdot w\!\left(\frac{\mu_{i}-\mu_{j}}{c}\right)\right]" display="block"><semantics><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo stretchy="false">←</mo><mrow><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mn>1</mn><mo>−</mo><mrow><mfrac><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><msup><mi>c</mi><mn>2</mn></msup></mfrac><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mpadded width="0.573em"><mi>w</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mfrac><mrow><msub><mi>μ</mi><mi>i</mi></msub><mo>−</mo><msub><mi>μ</mi><mi>j</mi></msub></mrow><mi>c</mi></mfrac><mo>)</mo></mrow></mrow></mrow></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.15)</span></div>

其中 <math alttext="c=\sqrt{2\beta^{2}+\sigma_{i}^{2}+\sigma_{j}^{2}}" display="inline"><semantics><mrow><mi>c</mi><mo>=</mo><msqrt><mrow><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><msup><mi>β</mi><mn>2</mn></msup></mrow><mo>+</mo><msubsup><mi>σ</mi><mi>i</mi><mn>2</mn></msubsup><mo>+</mo><msubsup><mi>σ</mi><mi>j</mi><mn>2</mn></msubsup></mrow></msqrt></mrow></semantics></math>，而 <math alttext="v(t)=\phi(t)/\Phi(t)" display="inline"><semantics><mrow><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>ϕ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>/</mo><mrow><mi mathvariant="normal">Φ</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></semantics></math>、<math alttext="w(t)=v(t)(v(t)+t)" display="inline"><semantics><mrow><mrow><mi>w</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>=</mo><mrow><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mrow><mi>v</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></mrow></mrow><mo>+</mo><mi>t</mi></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math> 是截断高斯修正因子（<math alttext="\phi" display="inline"><semantics><mi>ϕ</mi></semantics></math> 与 <math alttext="\Phi" display="inline"><semantics><mi mathvariant="normal">Φ</mi></semantics></math> 分别是标准正态分布的 PDF 与 CDF）。TrueSkill 的不确定性估计 <math alttext="\sigma_{i}" display="inline"><semantics><msub><mi>σ</mi><mi>i</mi></msub></semantics></math> 对于识别需要更多评估数据的模型尤为有用。

### 带置信区间的胜率

最简单的排序指标是胜率：在成对比较中模型 <math alttext="A" display="inline"><semantics><mi>A</mi></semantics></math> 被偏好的比例。给定 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 次比较中有 <math alttext="w" display="inline"><semantics><mi>w</mi></semantics></math> 次获胜，胜率为 <math alttext="\hat{p}=w/n" display="inline"><semantics><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>=</mo><mrow><mi>w</mi><mo>/</mo><mi>n</mi></mrow></mrow></semantics></math>。Wilson 得分置信区间 [380] 优于朴素的 Wald 区间，因为它在 <math alttext="p=0" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>0</mn></mrow></semantics></math> 与 <math alttext="p=1" display="inline"><semantics><mrow><mi>p</mi><mo>=</mo><mn>1</mn></mrow></semantics></math> 附近有更好的覆盖率：

<div class="hh-equation" id="ch14.e16"><math alttext="\text{CI}=\frac{\hat{p}+\frac{z^{2}}{2n}\pm z\sqrt{\frac{\hat{p}(1-\hat{p})}{n}+\frac{z^{2}}{4n^{2}}}}{1+\frac{z^{2}}{n}}" display="block"><semantics><mrow><mtext>CI</mtext><mo>=</mo><mfrac><mrow><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo>+</mo><mfrac><msup><mi>z</mi><mn>2</mn></msup><mrow><mn>2</mn><mo lspace="0em" rspace="0em">​</mo><mi>n</mi></mrow></mfrac></mrow><mo>±</mo><mrow><mi>z</mi><mo lspace="0em" rspace="0em">​</mo><msqrt><mrow><mfrac><mrow><mover accent="true"><mi>p</mi><mo>^</mo></mover><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mover accent="true"><mi>p</mi><mo>^</mo></mover></mrow><mo stretchy="false">)</mo></mrow></mrow><mi>n</mi></mfrac><mo>+</mo><mfrac><msup><mi>z</mi><mn>2</mn></msup><mrow><mn>4</mn><mo lspace="0em" rspace="0em">​</mo><msup><mi>n</mi><mn>2</mn></msup></mrow></mfrac></mrow></msqrt></mrow></mrow><mrow><mn>1</mn><mo>+</mo><mfrac><msup><mi>z</mi><mn>2</mn></msup><mi>n</mi></mfrac></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.16)</span></div>

其中 <math alttext="z=1.96" display="inline"><semantics><mrow><mi>z</mi><mo>=</mo><mn>1.96</mn></mrow></semantics></math> 对应 95% 区间。对于多方比较，胜率应针对一个固定基线模型计算，以确保可比性。

### Chatbot Arena 方法论

Chatbot Arena [433] 将上述要素组合成一个生产规模的评估系统：

<ol><li>用户提交 Prompt，并从两个匿名模型获得回答。</li><li>用户投票选出更偏好的回答（或判定平局）。</li><li>用 BT 模型聚合投票以生成排行榜。</li><li>为每个模型的分数报告自举置信区间。</li><li>置信区间重叠的模型被视为在统计上无法区分。</li></ol>

截至 2024 年，Chatbot Arena 已收集超过一百万条人类偏好投票，是最大的公开可用 LLM 偏好数据集。

## 生成任务的指标

生成指标用于量化在具有参考答案或明确正确性标准的任务上模型输出的质量。

### BLEU

BLEU（Bilingual Evaluation Understudy）[283] 度量假设 <math alttext="h" display="inline"><semantics><mi>h</mi></semantics></math> 与一个或多个参考 <math alttext="\mathcal{R}" display="inline"><semantics><mi>ℛ</mi></semantics></math> 之间的 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 精确率：

<div class="hh-equation" id="ch14.e17"><math alttext="\text{BLEU}=\text{BP}\cdot\exp\!\left(\sum_{n=1}^{N}w_{n}\log p_{n}\right)" display="block"><semantics><mrow><mtext>BLEU</mtext><mo>=</mo><mrow><mtext>BP</mtext><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mpadded width="1.370em"><mi>exp</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><munderover><mo lspace="0em" movablelimits="false">∑</mo><mrow><mi>n</mi><mo>=</mo><mn>1</mn></mrow><mi>N</mi></munderover><mrow><msub><mi>w</mi><mi>n</mi></msub><mo lspace="0.167em" rspace="0em">​</mo><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>p</mi><mi>n</mi></msub></mrow></mrow></mrow><mo>)</mo></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.17)</span></div>

其中 <math alttext="p_{n}" display="inline"><semantics><msub><mi>p</mi><mi>n</mi></msub></semantics></math> 是修正后的 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 精确率，<math alttext="w_{n}=1/N" display="inline"><semantics><mrow><msub><mi>w</mi><mi>n</mi></msub><mo>=</mo><mrow><mn>1</mn><mo>/</mo><mi>N</mi></mrow></mrow></semantics></math> 是均匀权重，BP 是长度惩罚（brevity penalty）：

<div class="hh-equation" id="ch14.e18"><math alttext="\text{BP}=\begin{cases}1&amp;\text{if }|h|&gt;|r|\\
e^{1-|r|/|h|}&amp;\text{if }|h|\leq|r|\end{cases}" display="block"><semantics><mrow><mtext>BP</mtext><mo>=</mo><mrow><mo>{</mo><mtable columnspacing="5pt" displaystyle="true" rowspacing="0pt"><mtr><mtd columnalign="left"><mn>1</mn></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><mi>h</mi><mo stretchy="false">|</mo></mrow></mrow><mo>&gt;</mo><mrow><mo stretchy="false">|</mo><mi>r</mi><mo stretchy="false">|</mo></mrow></mrow></mtd></mtr><mtr><mtd columnalign="left"><msup><mi>e</mi><mrow><mn>1</mn><mo>−</mo><mrow><mrow><mo stretchy="false">|</mo><mi>r</mi><mo stretchy="false">|</mo></mrow><mo>/</mo><mrow><mo stretchy="false">|</mo><mi>h</mi><mo stretchy="false">|</mo></mrow></mrow></mrow></msup></mtd><mtd columnalign="left"><mrow><mrow><mtext>if </mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">|</mo><mi>h</mi><mo stretchy="false">|</mo></mrow></mrow><mo>≤</mo><mrow><mo stretchy="false">|</mo><mi>r</mi><mo stretchy="false">|</mo></mrow></mrow></mtd></mtr></mtable></mrow></mrow></semantics></math><span class="hh-equation-number">(14.18)</span></div>

其中 <math alttext="|r|" display="inline"><semantics><mrow><mo stretchy="false">|</mo><mi>r</mi><mo stretchy="false">|</mo></mrow></semantics></math> 是最接近参考的长度。修正的 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 精确率将每个 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 的计数截断到其在任一参考中的最大计数：

<div class="hh-equation" id="ch14.e19"><math alttext="p_{n}=\frac{\sum_{\text{ngram}\in h}\min\!\left(\text{count}(\text{ngram},h),\,\max_{r\in\mathcal{R}}\text{count}(\text{ngram},r)\right)}{\sum_{\text{ngram}\in h}\text{count}(\text{ngram},h)}" display="block"><semantics><mrow><msub><mi>p</mi><mi>n</mi></msub><mo>=</mo><mfrac><mrow><msub><mo>∑</mo><mrow><mtext>ngram</mtext><mo>∈</mo><mi>h</mi></mrow></msub><mrow><mpadded width="1.653em"><mi>min</mi></mpadded><mo>⁡</mo><mrow><mo>(</mo><mrow><mtext>count</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mtext>ngram</mtext><mo>,</mo><mi>h</mi><mo stretchy="false">)</mo></mrow></mrow><mo rspace="0.337em">,</mo><mrow><mrow><msub><mi>max</mi><mrow><mi>r</mi><mo>∈</mo><mi>ℛ</mi></mrow></msub><mo lspace="0.167em">⁡</mo><mtext>count</mtext></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mtext>ngram</mtext><mo>,</mo><mi>r</mi><mo stretchy="false">)</mo></mrow></mrow><mo>)</mo></mrow></mrow></mrow><mrow><msub><mo>∑</mo><mrow><mtext>ngram</mtext><mo>∈</mo><mi>h</mi></mrow></msub><mrow><mtext>count</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mtext>ngram</mtext><mo>,</mo><mi>h</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.19)</span></div>

### ROUGE

ROUGE（Recall-Oriented Understudy for Gisting Evaluation）[217] 是一族面向摘要、以召回率为导向的指标：



其中 LCS 表示最长公共子序列。ROUGE-1 与 ROUGE-2 度量 unigram 与 bigram 召回率；ROUGE-L 捕捉句子级结构。F-measure 变体则平衡精确率与召回率：

<div class="hh-equation" id="ch14.e22"><math alttext="\text{ROUGE-N}_{F}=\frac{(1+\beta^{2})\cdot P\cdot R}{\beta^{2}P+R}" display="block"><semantics><mrow><msub><mtext>ROUGE-N</mtext><mi>F</mi></msub><mo>=</mo><mfrac><mrow><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>+</mo><msup><mi>β</mi><mn>2</mn></msup></mrow><mo rspace="0.055em" stretchy="false">)</mo></mrow><mo rspace="0.222em">⋅</mo><mi>P</mi><mo lspace="0.222em" rspace="0.222em">⋅</mo><mi>R</mi></mrow><mrow><mrow><msup><mi>β</mi><mn>2</mn></msup><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow><mo>+</mo><mi>R</mi></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.22)</span></div>

其中 <math alttext="\beta=1" display="inline"><semantics><mrow><mi>β</mi><mo>=</mo><mn>1</mn></mrow></semantics></math> 表示等权重。

### BERTScore

BERTScore [426] 使用预训练 BERT 模型的上下文嵌入计算 token 级相似度。给定假设 token <math alttext="\hat{\mathbf{x}}=\langle\hat{x}_{1},\ldots,\hat{x}_{m}\rangle" display="inline"><semantics><mrow><mover accent="true"><mi>𝐱</mi><mo>^</mo></mover><mo>=</mo><mrow><mo stretchy="false">⟨</mo><mrow><msub><mover accent="true"><mi>x</mi><mo>^</mo></mover><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mover accent="true"><mi>x</mi><mo>^</mo></mover><mi>m</mi></msub></mrow><mo stretchy="false">⟩</mo></mrow></mrow></semantics></math> 与参考 token <math alttext="\mathbf{x}=\langle x_{1},\ldots,x_{n}\rangle" display="inline"><semantics><mrow><mi>𝐱</mi><mo>=</mo><mrow><mo stretchy="false">⟨</mo><mrow><msub><mi>x</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>x</mi><mi>n</mi></msub></mrow><mo stretchy="false">⟩</mo></mrow></mrow></semantics></math>，其嵌入分别为 <math alttext="\hat{\mathbf{e}}_{i}" display="inline"><semantics><msub><mover accent="true"><mi>𝐞</mi><mo>^</mo></mover><mi>i</mi></msub></semantics></math> 与 <math alttext="\mathbf{e}_{j}" display="inline"><semantics><msub><mi>𝐞</mi><mi>j</mi></msub></semantics></math>：



BERTScore 与人类判断的相关性优于 BLEU 和 ROUGE，尤其是对于改写以及语义等价但用词不同的输出。使用逆文档频率（IDF）进行重要性加权可进一步提升相关性：

<div class="hh-equation" id="ch14.e26"><math alttext="R_{\text{BERT}}^{\text{idf}}=\frac{\sum_{x_{j}\in\mathbf{x}}\text{idf}(x_{j})\max_{\hat{x}_{i}}\cos(\hat{\mathbf{e}}_{i},\mathbf{e}_{j})}{\sum_{x_{j}\in\mathbf{x}}\text{idf}(x_{j})}" display="block"><semantics><mrow><msubsup><mi>R</mi><mtext>BERT</mtext><mtext>idf</mtext></msubsup><mo>=</mo><mfrac><mrow><msub><mo>∑</mo><mrow><msub><mi>x</mi><mi>j</mi></msub><mo>∈</mo><mi>𝐱</mi></mrow></msub><mrow><mtext>idf</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow><mo lspace="0.167em" rspace="0em">​</mo><mrow><msub><mi>max</mi><msub><mover accent="true"><mi>x</mi><mo>^</mo></mover><mi>i</mi></msub></msub><mo lspace="0.167em">⁡</mo><mrow><mi>cos</mi><mo>⁡</mo><mrow><mo stretchy="false">(</mo><msub><mover accent="true"><mi>𝐞</mi><mo>^</mo></mover><mi>i</mi></msub><mo>,</mo><msub><mi>𝐞</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow></mrow><mrow><msub><mo>∑</mo><mrow><msub><mi>x</mi><mi>j</mi></msub><mo>∈</mo><mi>𝐱</mi></mrow></msub><mrow><mtext>idf</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msub><mi>x</mi><mi>j</mi></msub><mo stretchy="false">)</mo></mrow></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.26)</span></div>

### METEOR

METEOR [19] 通过对 unigram 匹配计算 F-score 来解决 BLEU 的召回盲区，并配有词干化与同义词匹配等额外模块：

<div class="hh-equation" id="ch14.e27"><math alttext="\text{METEOR}=F_{\text{mean}}\cdot(1-\text{Pen})" display="block"><semantics><mrow><mtext>METEOR</mtext><mo>=</mo><mrow><msub><mi>F</mi><mtext>mean</mtext></msub><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo stretchy="false">(</mo><mrow><mn>1</mn><mo>−</mo><mtext>Pen</mtext></mrow><mo stretchy="false">)</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.27)</span></div>

其中 <math alttext="F_{\text{mean}}=\frac{10PR}{R+9P}" display="inline"><semantics><mrow><msub><mi>F</mi><mtext>mean</mtext></msub><mo>=</mo><mfrac><mrow><mn>10</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi><mo lspace="0em" rspace="0em">​</mo><mi>R</mi></mrow><mrow><mi>R</mi><mo>+</mo><mrow><mn>9</mn><mo lspace="0em" rspace="0em">​</mo><mi>P</mi></mrow></mrow></mfrac></mrow></semantics></math>（召回率加权的调和平均），而碎片化惩罚 <math alttext="\text{Pen}=0.5\cdot(c/u_{m})^{3}" display="inline"><semantics><mrow><mtext>Pen</mtext><mo>=</mo><mrow><mn>0.5</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><msup><mrow><mo stretchy="false">(</mo><mrow><mi>c</mi><mo>/</mo><msub><mi>u</mi><mi>m</mi></msub></mrow><mo stretchy="false">)</mo></mrow><mn>3</mn></msup></mrow></mrow></semantics></math> 会惩罚非连续的匹配（<math alttext="c" display="inline"><semantics><mi>c</mi></semantics></math> = 片段数，<math alttext="u_{m}" display="inline"><semantics><msub><mi>u</mi><mi>m</mi></msub></semantics></math> = 匹配的 unigram 数）。

### 困惑度（Perplexity）

困惑度度量语言模型对留出文本序列 <math alttext="w_{1},w_{2},\ldots,w_{T}" display="inline"><semantics><mrow><msub><mi>w</mi><mn>1</mn></msub><mo>,</mo><msub><mi>w</mi><mn>2</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><msub><mi>w</mi><mi>T</mi></msub></mrow></semantics></math> 的预测能力：

<div class="hh-equation" id="ch14.e28"><math alttext="\text{PPL}(w_{1:T})=\exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P_{\theta}(w_{t}\mid w_{1:t-1})\right)" display="block"><semantics><mrow><mtext>PPL</mtext><mrow><mo stretchy="false">(</mo><msub><mi>w</mi><mrow><mn>1</mn><mo lspace="0.278em" rspace="0.278em">:</mo><mi>T</mi></mrow></msub><mo stretchy="false">)</mo></mrow><mo>=</mo><mpadded width="1.370em"><mi>exp</mi></mpadded><mrow><mo>(</mo><mo lspace="0em">−</mo><mfrac><mn>1</mn><mi>T</mi></mfrac><munderover><mo movablelimits="false">∑</mo><mrow><mi>t</mi><mo>=</mo><mn>1</mn></mrow><mi>T</mi></munderover><mi>log</mi><msub><mi>P</mi><mi>θ</mi></msub><mrow><mo stretchy="false">(</mo><msub><mi>w</mi><mi>t</mi></msub><mo lspace="0em" rspace="0.167em">∣</mo><msub><mi>w</mi><mrow><mn>1</mn><mo lspace="0.278em" rspace="0.278em">:</mo><mrow><mi>t</mi><mo>−</mo><mn>1</mn></mrow></mrow></msub><mo stretchy="false">)</mo></mrow><mo>)</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(14.28)</span></div>

困惑度越低表示预测性能越好。困惑度适用于在相同分词与测试集上比较模型，但对于词表或分词器不同的模型并不可直接比较。就评估而言，困惑度最有价值的用途是作为健全性检查以及检测分布漂移。

### 代码任务的 Pass@k

对于代码生成，功能正确性通过将生成的代码在测试用例上执行来衡量。pass@<math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 指标 [47] 估计 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个生成样本中至少有一个通过全部测试的概率：

<div class="hh-equation" id="ch14.e29"><math alttext="\text{pass@}k=\mathbb{E}_{\text{problems}}\!\left[1-\frac{\binom{n-c}{k}}{\binom{n}{k}}\right]" display="block"><semantics><mrow><mrow><mtext>pass@</mtext><mo lspace="0em" rspace="0em">​</mo><mi>k</mi></mrow><mo>=</mo><mrow><msub><mi>𝔼</mi><mtext>problems</mtext></msub><mo lspace="0em" rspace="0em">​</mo><mrow><mo>[</mo><mrow><mn>1</mn><mo>−</mo><mfrac><mrow><mo>(</mo><mfrac linethickness="0pt"><mrow><mi>n</mi><mo>−</mo><mi>c</mi></mrow><mi>k</mi></mfrac><mo>)</mo></mrow><mrow><mo>(</mo><mfrac linethickness="0pt"><mi>n</mi><mi>k</mi></mfrac><mo>)</mo></mrow></mfrac></mrow><mo>]</mo></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.29)</span></div>

其中 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math> 是每个问题生成的样本总数，<math alttext="c" display="inline"><semantics><mi>c</mi></semantics></math> 是通过的样本数。这个无偏估计量避免了朴素估计量的高方差（朴素做法是恰好采样 <math alttext="k" display="inline"><semantics><mi>k</mi></semantics></math> 个解并检查是否有任何一个通过）。实践中会生成 <math alttext="n=200" display="inline"><semantics><mrow><mi>n</mi><mo>=</mo><mn>200</mn></mrow></semantics></math> 个样本，并报告 pass@1、pass@10、pass@100。

### 精确匹配与 F1

对于抽取式问答（例如 SQuAD），两个标准指标是：

<ul><li>精确匹配（EM）：在归一化（转为小写、去除冠词与标点）之后，预测答案字符串是否与任一金标答案完全一致的二元指示量。</li><li>Token 级 F1：将预测与金标答案视为 token 袋，并计算 F1 分数：<math alttext="F1=\frac{2\cdot|\text{pred}\cap\text{gold}|}{|\text{pred}|+|\text{gold}|}" display="block"><semantics><mrow><mrow><mi>F</mi><mo lspace="0em" rspace="0em">​</mo><mn>1</mn></mrow><mo>=</mo><mfrac><mrow><mn>2</mn><mo lspace="0.222em" rspace="0.222em">⋅</mo><mrow><mo stretchy="false">|</mo><mrow><mtext>pred</mtext><mo>∩</mo><mtext>gold</mtext></mrow><mo stretchy="false">|</mo></mrow></mrow><mrow><mrow><mo stretchy="false">|</mo><mtext>pred</mtext><mo stretchy="false">|</mo></mrow><mo>+</mo><mrow><mo stretchy="false">|</mo><mtext>gold</mtext><mo stretchy="false">|</mo></mrow></mrow></mfrac></mrow></semantics></math><span class="hh-tag">(14.30)</span></li></ul>

在多答案设置中，报告对全部金标答案取得的最大 F1。

<div class="hh-table" id="ch14.t3"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>指标</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>任务</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>是否需要参考？</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>与人类判断的相关性</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">BLEU</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>翻译</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低–中</span></span></td></tr><tr><th class="ltx_align_left">ROUGE</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>摘要</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中</span></span></td></tr><tr><th class="ltx_align_left">BERTScore</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>通用 NLG</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td></tr><tr><th class="ltx_align_left">METEOR</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>翻译</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>中–高</span></span></td></tr><tr><th class="ltx_align_left">困惑度</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>语言模型质量</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>是</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>低</span></span></td></tr><tr><th class="ltx_align_left">Pass@k</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>代码生成</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否（测试）</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极高</span></span></td></tr><tr><th class="ltx_align_left">精确匹配</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>抽取式问答</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>极高</span></span></td></tr><tr><th class="ltx_align_left">Token F1</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>抽取式问答</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>否</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>高</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 14.3：</span>生成指标汇总：适用性与关键性质。</p></div>

## Agent 任务的指标

Agentic LLM 在环境中运行、执行动作序列，并且必须完成多步任务。标准的生成指标并不充分；面向智能体的评估需要能够刻画任务完成度、效率以及中间步骤质量的指标。

### 任务成功率

智能体任务的首要指标是任务成功率（TSR）：智能体达成既定目标状态的任务所占比例：

<div class="hh-equation" id="ch14.e31"><math alttext="\text{TSR}=\frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\mathbf{1}[\text{goal}(\tau)\text{ achieved}]" display="block"><semantics><mrow><mtext>TSR</mtext><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>𝒯</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>τ</mi><mo>∈</mo><mi>𝒯</mi></mrow></munder><mrow><mn>𝟏</mn><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><mtext>goal</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow><mo lspace="0em" rspace="0em">​</mo><mtext> achieved</mtext></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.31)</span></div>

目标达成通常由确定性的 oracle 验证（例如检查数据库状态、文件系统状态或测试用例执行结果）。对于支持部分得分的任务，可以定义分级成功指标：

<div class="hh-equation" id="ch14.e32"><math alttext="\text{TSR}_{\text{graded}}=\frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\text{score}(\tau)\in[0,1]" display="block"><semantics><mrow><msub><mtext>TSR</mtext><mtext>graded</mtext></msub><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>𝒯</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>τ</mi><mo>∈</mo><mi>𝒯</mi></mrow></munder><mrow><mtext>score</mtext><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>τ</mi><mo stretchy="false">)</mo></mrow></mrow></mrow></mrow><mo>∈</mo><mrow><mo stretchy="false">[</mo><mrow><mn>0</mn><mo>,</mo><mn>1</mn></mrow><mo stretchy="false">]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(14.32)</span></div>

### 轨迹效率

成功的智能体（Agent）应在尽量少的非必要动作下完成任务。轨迹效率衡量最优轨迹长度与智能体实际轨迹长度之比：

<div class="hh-equation" id="ch14.e33"><math alttext="\eta=\frac{L^{*}}{L_{\text{agent}}}" display="block"><semantics><mrow><mi>η</mi><mo>=</mo><mfrac><msup><mi>L</mi><mo>∗</mo></msup><msub><mi>L</mi><mtext>agent</mtext></msub></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.33)</span></div>

其中 <math alttext="L^{*}" display="inline"><semantics><msup><mi>L</mi><mo>∗</mo></msup></semantics></math> 是最短成功轨迹的长度（由 oracle 或人类专家计算），<math alttext="L_{\text{agent}}" display="inline"><semantics><msub><mi>L</mi><mtext>agent</mtext></msub></semantics></math> 是智能体采取的动作数。<math alttext="\eta\in(0,1]" display="inline"><semantics><mrow><mi>η</mi><mo>∈</mo><mrow><mo stretchy="false">(</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></mrow></semantics></math>，<math alttext="\eta=1" display="inline"><semantics><mrow><mi>η</mi><mo>=</mo><mn>1</mn></mrow></semantics></math> 表示达到最优效率。对于失败的轨迹，<math alttext="\eta=0" display="inline"><semantics><mrow><mi>η</mi><mo>=</mo><mn>0</mn></mrow></semantics></math>。

一个互补指标是 <em>冗余率（redundancy rate）</em>：即未出现在任何最优轨迹中的智能体动作所占比例。

### 工具调用准确性

对于调用外部工具（API、代码解释器、搜索引擎）的智能体，工具使用准确率衡量工具调用的正确性：

<div class="hh-equation" id="ch14.e34"><math alttext="\text{TUA}=\frac{\text{\# correct tool calls}}{\text{\# total tool calls}}" display="block"><semantics><mrow><mtext>TUA</mtext><mo>=</mo><mfrac><mtext># correct tool calls</mtext><mtext># total tool calls</mtext></mfrac></mrow></semantics></math><span class="hh-equation-number">(14.34)</span></div>

一次工具调用正确需满足：(a) 选中了正确的工具，(b) 参数有效，(c) 调用发生在轨迹中恰当的位置。对于工具选择正确但参数有误的情况，可以给予部分得分。

### 多步推理准确性

对于需要推理链的任务（例如多跳问答、数学问题求解），步骤级准确率衡量正确推理步骤所占比例：

<div class="hh-equation" id="ch14.e35"><math alttext="\text{SRA}=\frac{1}{|\mathcal{T}|}\sum_{\tau\in\mathcal{T}}\frac{1}{|S_{\tau}|}\sum_{s\in S_{\tau}}\mathbf{1}[s\text{ is correct}]" display="block"><semantics><mrow><mtext>SRA</mtext><mo>=</mo><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><mi>𝒯</mi><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>τ</mi><mo>∈</mo><mi>𝒯</mi></mrow></munder><mrow><mfrac><mn>1</mn><mrow><mo stretchy="false">|</mo><msub><mi>S</mi><mi>τ</mi></msub><mo stretchy="false">|</mo></mrow></mfrac><mo lspace="0em" rspace="0em">​</mo><mrow><munder><mo movablelimits="false">∑</mo><mrow><mi>s</mi><mo>∈</mo><msub><mi>S</mi><mi>τ</mi></msub></mrow></munder><mrow><mn>𝟏</mn><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">[</mo><mrow><mi>s</mi><mo lspace="0em" rspace="0em">​</mo><mtext> is correct</mtext></mrow><mo stretchy="false">]</mo></mrow></mrow></mrow></mrow></mrow></mrow></mrow></semantics></math><span class="hh-equation-number">(14.35)</span></div>

其中 <math alttext="S_{\tau}" display="inline"><semantics><msub><mi>S</mi><mi>τ</mi></msub></semantics></math> 是轨迹 <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math> 中推理步骤的集合。步骤正确性可由过程奖励模型（PRM）或人工标注来验证。

### SWE-bench 方法论

SWE-bench [169] 在真实软件工程任务上评估 LLM：给定一个 GitHub issue 描述与仓库代码库，模型必须生成解决该 issue 的 patch。评估流程如下：

<ol><li>将 issue 描述与相关代码上下文提供给模型。</li><li>模型生成一个 patch（统一 diff 格式）。</li><li>将 patch 应用到仓库。</li><li>执行仓库的测试套件；若全部测试通过，则任务成功。</li></ol>

主要指标为 % Resolved：生成的 patch 能通过全部测试的 issue 所占比例。SWE-bench Verified 是一个经过人工标注者验证为可解且无歧义的 500 题精选子集。SWE-bench Lite 则是一个 300 题的子集，专为更快的评估而设计。

### WebArena 方法论

WebArena [440] 在沙箱浏览器环境中，以贴近真实的网页导航任务评估智能体。该基准包含跨五个 Web 应用（电商、社交论坛、协作开发、内容管理与地图）的 812 个任务。评估方式：

<ul><li>功能性评估：通过检查应用状态来验证任务结果（例如“商品是否已加入购物车？”“帖子是否已创建？”）。</li><li>基于 URL 的评估：对于导航任务，将最终 URL 与期望 URL 进行比较。</li><li>基于程序的评估：由自定义评估脚本检查复杂条件（例如“价格是否小于 $50？”）。</li></ul>

主要指标是任务成功率。人类表现约为 78%；当前最先进的智能体约为 35–45%。

<div class="hh-table" id="ch14.t4"><table class="ltx_tabular ltx_align_middle"><thead><tr><th class="ltx_align_left"><span>基准</span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>领域</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>任务数</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>评估方式</span></span></span></th><th class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><span>SOTA（%）</span></span></span></th></tr></thead><tbody><tr><th class="ltx_align_left">SWE-bench</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>软件工程</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>2,294</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>测试执行</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>43</span></span></td></tr><tr><th class="ltx_align_left">SWE-bench Lite</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>软件工程</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>300</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>测试执行</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>50</span></span></td></tr><tr><th class="ltx_align_left">WebArena</th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>网页导航</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>812</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>状态/URL/程序</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>40</span></span></td></tr><tr><th class="ltx_align_left">ALFWorld <span>[329]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>家居任务</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>3,553</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>仿真器状态</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>90</span></span></td></tr><tr><th class="ltx_align_left">AgentBench <span>[230]</span></th><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>多领域</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>1,091</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span>任务特定</span></span></td><td class="ltx_align_left ltx_align_top"><span class="ltx_align_top"><span><math alttext="\sim" display="inline"><semantics><mo>∼</mo><span></span></semantics></math>45</span></span></td></tr></tbody></table><p class="hh-table-caption"><span class="hh-tag">表 14.4：</span>智能体评估基准对比。</p></div>

## LLM-as-Judge

LLM 作为评判者（LLM-as-judge）[433] 使用一个能力强的 LLM 来评估其他（或同一个）LLM 的输出。该方法无需人工标注就能扩展到大规模评估集合，并能为其判断提供详尽的理由。

### 设置与 Prompt 模板

评判者会收到一个 Prompt、一个或多个模型回答，以及一份评估细则。常见的三种格式：

评判者对单个回答给出绝对分数：

评判者比较两个回答并选出更优者：

评判者会获得一个参考答案，并据此对回答进行打分。这对于评判者本身可能缺乏可靠知识的事实型任务尤其有用。

### 位置偏置的缓解

LLM 评判者会表现出 <em>位置偏置（position bias）</em>：系统性地偏好出现在特定位置（首位或末位）的回答。此偏置可能高达 10–15 个百分点。缓解策略：

<ol><li>交换增强：对每一对回答都以两种顺序（A vs. B 与 B vs. A）进行评判。判断一致则采纳；判断不一致则记为平局。</li><li>校准式 Prompt：显式指令评判者：“你的评估不应受到回答呈现顺序的影响。”</li><li>评判者集成：使用多个评判者并配以不同的位置顺序，再聚合其裁决。</li><li>强制思维链：要求评判者在给出裁决前先产出详尽理由，以降低对表层位置线索的依赖。</li></ol>

### 多评判者面板

单一评判者可能带有系统性偏置。来自不同模型家族的评判者面板能提供更稳健的评估。给定 <math alttext="J" display="inline"><semantics><mi>J</mi></semantics></math> 位评判者，其裁决为 <math alttext="v_{1},\ldots,v_{J}\in\{A,B,\text{tie}\}" display="inline"><semantics><mrow><msub><mi>v</mi><mn>1</mn></msub><mo>,</mo><mi mathvariant="normal">…</mi><mo>,</mo><mrow><msub><mi>v</mi><mi>J</mi></msub><mo>∈</mo><mrow><mo stretchy="false">{</mo><mi>A</mi><mo>,</mo><mi>B</mi><mo>,</mo><mtext>tie</mtext><mo stretchy="false">}</mo></mrow></mrow></mrow></semantics></math>，面板裁决由多数投票决定。面板一致率为：

<div class="hh-equation" id="ch14.e36"><math alttext="\text{Agreement}=\frac{1}{\binom{J}{2}}\sum_{i&lt;j}\mathbf{1}[v_{i}=v_{j}]" display="block"><semantics><mrow><mtext>Agreement</mtext><mo>=</mo><mfrac><mn>1</mn><mrow><mo>(</mo><mfrac linethickness="0pt"><mi>J</mi><mn>2</mn></mfrac><mo>)</mo></mrow></mfrac><munder><mo movablelimits="false">∑</mo><mrow><mi>i</mi><mo>&lt;</mo><mi>j</mi></mrow></munder><mn>𝟏</mn><mrow><mo stretchy="false">[</mo><msub><mi>v</mi><mi>i</mi></msub><mo>=</mo><msub><mi>v</mi><mi>j</mi></msub><mo stretchy="false">]</mo></mrow></mrow></semantics></math><span class="hh-equation-number">(14.36)</span></div>

对于三人评判者面板，一致裁决（三人全部同意）视为高置信度；2–1 分裂视为中置信度；三方平局视为低置信度。

### LLM 评判者的一致性指标

为验证 LLM 评判者，需将其裁决与留出集上的人工标注进行比较。关键指标：

<ul><li>一致率：评判者与人类判断一致的条目所占比例。</li><li>Cohen 的 <math alttext="\kappa" display="inline"><semantics><mi>κ</mi></semantics></math>：经机会校正的一致性（式 ）。</li><li>Spearman 的 <math alttext="\rho" display="inline"><semantics><mi>ρ</mi></semantics></math>：评判者分数与人类分数之间的秩相关，适用于有序评分。</li><li>Kendall 的 <math alttext="\tau" display="inline"><semantics><mi>τ</mi></semantics></math>：另一种秩相关，对并列情况更稳健。</li></ul>

若评判者在具有代表性的样本上相对人工标注者达到 <math alttext="\kappa&gt;0.6" display="inline"><semantics><mrow><mi>κ</mi><mo>&gt;</mo><mn>0.6</mn></mrow></semantics></math>，且一致率为 <math alttext="&gt;80\%" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mrow><mn>80</mn><mo>%</mo></mrow></mrow></semantics></math>，则认为该评判者可靠。

### G-Eval 框架

G-Eval [232] 是一个基于 LLM 的结构化评估框架，它使用思维链 Prompt 与 Token 概率加权来产生更可靠的分数。该框架：

<ol><li>生成评估步骤：让 LLM 为评估任务生成详细的评估细则（例如“列出你会用来评估摘要连贯性的步骤”）。</li><li>基于概率加权打分：对每个分数取值 <math alttext="s\in\{1,2,3,4,5\}" display="inline"><semantics><mrow><mi>s</mi><mo>∈</mo><mrow><mo stretchy="false">{</mo><mrow><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mn>3</mn><mo>,</mo><mn>4</mn><mo>,</mo><mn>5</mn></mrow><mo stretchy="false">}</mo></mrow></mrow></semantics></math>，从评判者模型获取对数概率 <math alttext="\log P_{\theta}(s\mid\text{prompt, steps, response})" display="inline"><semantics><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo fence="true" lspace="0em" rspace="0em">∣</mo><mtext>prompt, steps, response</mtext><mo stretchy="false">)</mo></mrow></mrow></semantics></math>。最终分数为概率加权平均：<math alttext="\text{G-Eval score}=\sum_{s=1}^{5}s\cdot\frac{e^{\log P_{\theta}(s)}}{\sum_{s^{\prime}=1}^{5}e^{\log P_{\theta}(s^{\prime})}}" display="block"><semantics><mrow><mtext>G-Eval score</mtext><mo rspace="0.111em">=</mo><mrow><mrow><munderover><mo movablelimits="false">∑</mo><mrow><mi>s</mi><mo>=</mo><mn>1</mn></mrow><mn>5</mn></munderover><mi>s</mi></mrow><mo lspace="0.222em" rspace="0.222em">⋅</mo><mfrac><msup><mi>e</mi><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><mi>s</mi><mo stretchy="false">)</mo></mrow></mrow></msup><mrow><msubsup><mo>∑</mo><mrow><msup><mi>s</mi><mo>′</mo></msup><mo>=</mo><mn>1</mn></mrow><mn>5</mn></msubsup><msup><mi>e</mi><mrow><mrow><mi>log</mi><mo lspace="0.167em">⁡</mo><msub><mi>P</mi><mi>θ</mi></msub></mrow><mo lspace="0em" rspace="0em">​</mo><mrow><mo stretchy="false">(</mo><msup><mi>s</mi><mo>′</mo></msup><mo stretchy="false">)</mo></mrow></mrow></msup></mrow></mfrac></mrow></mrow></semantics></math><span class="hh-tag">(14.37)</span></li><li>归一化：用最大分数相除，把分数映射到 <math alttext="[0,1]" display="inline"><semantics><mrow><mo stretchy="false">[</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo stretchy="false">]</mo></mrow></semantics></math>。</li></ol>

相比直接 Prompt，G-Eval 与人类判断的相关性更高，对于连贯性与一致性这类细微维度尤其如此，因为概率加权捕捉的是评判者的不确定性，而不是强行做出离散选择。

## 评估陷阱

即便精心设计的评估流水线也可能产生误导性结果。本节梳理最常见的失败模式。

### 基准污染

基准污染（Benchmark Contamination）指评估数据出现在模型训练集中，无论是直接出现（逐字包含）还是间接出现（改写或语义相似的内容）。被污染的模型会获得虚高的分数，这些分数并不反映真实的泛化能力。

<ul><li><math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 重叠：计算与训练语料高度 <math alttext="n" display="inline"><semantics><mi>n</mi></semantics></math>-gram 重叠（例如 ROUGE-L <math alttext="&gt;0.8" display="inline"><semantics><mrow><mphantom></mphantom><mo>&gt;</mo><mn>0.8</mn></mrow></semantics></math>）的评估样本所占比例。</li><li>成员推断：使用成员推断攻击来估计每个评估样本曾出现在训练集中的概率。</li><li>金丝雀串：在评估样本中嵌入唯一的随机生成字符串，并检查模型能否补全它们。</li><li>时间留出：使用在模型训练截止日期之后创建的评估数据。</li></ul>

<ul><li>维护一个从不公开发布的私有测试集。</li><li>定期用新样本刷新基准。</li><li>报告训练数据截止日期与去污染流程。</li></ul>

### 对基准的过拟合

即便没有直接污染，模型也可能通过反复评估与超参数调优被隐式地针对特定基准优化。这是<em>自适应过拟合</em>的一种形式：基准把信息泄漏进了模型开发决策。

### 评估中的 Goodhart 定律

古德哈特定律（Goodhart's Law）指出：<em>“当一种度量成为目标时，它就不再是一种好的度量。”</em>[112] 在 LLM 评估中，这有多种表现：

<ul><li>奖励作弊（reward hacking）：用 RLHF 训练的模型会学会利用奖励模型，而不是真正地改进。模型可能学会生成冗长、听起来自信、在奖励模型上得分很高但事实上错误的回答。</li><li>指标博弈：为最大化 BLEU 或 ROUGE 而微调的模型，可能产生在这些指标上得分很高但对人类不太有用的输出。</li><li>评判者博弈：使用 LLM 作为评判者反馈训练的模型，可能学会评判者的偏置（例如冗长偏置），而不是真正提升质量。</li></ul>

### 其他陷阱

评估 Prompt 的细微改动（例如加入“Think step by step”或改变答案格式）都可能让 LLM 的表现出现剧烈变化。务必报告所用 Prompt 的确切内容，并考虑在多个 Prompt 变体上进行评估。

对不同难度与分数分布的任务求分数平均，可能产生误导性的聚合指标。在简单任务上表现出色但在困难任务上失败的模型，其平均分可能与表现均匀的模型相同。

人工评估者并不是最终用户的随机样本。众包平台上的标注者在偏好、文化背景与领域知识上可能与目标用户群体不同。

评估 Prompt 往往比真实用户查询更短、更整洁、更规范。在基准 Prompt 上表现良好的模型，在生产环境中出现的含噪、歧义、多轮对话上可能显著退化。

<aside class="hh-next">
<p class="hh-next-label">下一篇</p>
<p class="hh-next-title">第 15 章 智能体 AI 简介</p>
<p class="hh-next-hint">在「智能体 AI 漫游指南」分类下继续阅读。</p>
</aside>

</div>

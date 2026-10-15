# Online Forecasting, Model Adaptation, and Configuration Selection Papers


| Venue        | Paper                                                        | Corresp. Author                | Key Words                                                    | Links                                                        |
| ------------ | ------------------------------------------------------------ | ------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ICLR 2023    | **Learning Fast and Slow for Online Time Series Forecasting** | Quang Pham                     | Fast-Slow Adaptation, Associative Memory, Catastrophic Forgetting | [Paper](https://openreview.net/forum?id=q-PbpHD3EOk), [Code](https://github.com/salesforce/fsnet) |
| NeurIPS 2023 | **OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling** | Qingsong Wen                   | Online Ensembling, Adaptive Weighting, Cross-Variable Modeling | [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract.html), [Code](https://github.com/yfzhang114/OneNet) |
| ICLR 2025    | **Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage** | Ying-yee Ava Lau               | Fast-Slow Adaptation, Experience Replay, Delayed Feedback, Information Leakage | [Paper](https://proceedings.iclr.cc/paper_files/paper/2025/hash/46e624c244cff669223d488defd4e835-Abstract-Conference.html), [Code](https://github.com/yyalau/iclr2025_dsof) |
| ICML 2025    | **Lightweight Online Adaption for Time Series Foundation Model Forecasts** | Thomas L. Lee*; William Toner* | Foundation Model Adaptation, Online Ensembling, Lightweight Adaptation | [Paper](https://proceedings.mlr.press/v267/lee25ag.html) |
| KDD 2025     | **Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting** | Yanyan Shen                    | Proactive Adaptation, Delayed Feedback, Parameter Adaptation | [Paper](https://arxiv.org/abs/2412.08435), [Code](https://github.com/SJTU-DMTai/OnlineTSF) |
| IJCAI 2025   | **Learning to Extrapolate and Adjust: Two-Stage Meta-Learning for Concept Drift in Online Time Series Forecasting** | Weiqi Chen                     | Meta-Learning, Drift Decomposition, Latent-Space Adaptation | [Paper](https://www.ijcai.org/proceedings/2025/542), [Code](https://github.com/vc12301/LEAF) |
| WWW 2026     | **Dynamic Multi-period Experts for Online Time Series Forecasting** | Hwanjo Yu                      | Mixture-of-Experts, Drift Decomposition, Periodic Pattern Retrieval, Uncertainty-Aware Routing | [Paper](https://arxiv.org/abs/2603.09062), [Code](https://github.com/shhong97/DynaME) |
| ICLR 2026    | **Online Time Series Prediction Using Feature Adjustment**   | Chao Yang                      | Feature Adaptation, Historical Gradients, Delayed Feedback, Lightweight Adaptation | [Paper](https://arxiv.org/abs/2509.03810), [Code](https://github.com/xiannanhuang/ADAPT-Z) |
| ICLR 2026    | **When to Retrain after Drift: A Data-Only Test of Post-Drift Data Size Sufficiency** | Ren Fujiwara                   | Retraining Decision, Data Sufficiency, Data-Only Detection | [Paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d5ff2b092d81618a743c15cf38f8d134-Abstract-Conference.html) |
| ICML 2025    | **When to retrain a machine learning model**                 | Florence Regol                 | Retraining Decision, Uncertainty-Aware Decision, Cost-Aware Adaptation | [Paper](https://proceedings.mlr.press/v267/regol25a.html) |
| KDD 2026     | **Online Irregular Multivariate Time Series Forecasting via Uncertainty-Driven Dual-Expert Calibration** | Songhe Feng                    | Dual-Expert Calibration, Uncertainty-Aware Routing, Irregular Sampling, Lightweight Adaptation | [Paper](https://arxiv.org/abs/2605.28603), [Code](https://github.com/HaonanWen/Under-Cali) |
| IJCAI 2026   | **Beyond Uniform Updates: Drift Pattern Aware Online Time Series Forecasting Under Delayed Feedback** | Xingwang Li                    | Drift-Aware Update, Delayed Feedback, Selective Adaptation, Drift Retrieval | [Paper](https://www.ijcai.org/proceedings/2026/) |
| arXiv 2026   | **Natural Language Query to Configuration for Retrieval Agents** | Melissa Z. Pan    | Per-Query Configuration, Retrieval Agents, Workload-Specific Features, Per-Configuration Predictors, Cost–Quality Trade-off | [Paper](https://arxiv.org/abs/2605.27361), [HTML](https://arxiv.org/html/2605.27361v1) |

# Concept Drift

## Drift Detection

| Venue | Paper | Corresp. Author | Key Words | Links |
| --- | --- | --- | --- | --- |
| Biometrika 1954 | **Continuous Inspection Schemes** | E. S. Page (sole author) | Sequential Change Detection, CUSUM, Page-Hinkley | [Paper](https://doi.org/10.1093/biomet/41.1-2.100) |
| Technometrics 1959 | **Control Chart Tests Based on Geometric Moving Averages** | S. W. Roberts (sole author) | Geometric Moving Average, Statistical Process Control, GMA | [Paper](https://doi.org/10.1080/00401706.1959.10489860) |
| SBIA 2004 | **Learning with Drift Detection** | — | Error-Rate Monitoring, Statistical Warning, Drift Detection Method, DDM | [Paper](https://doi.org/10.1007/978-3-540-28645-5_29) |
| KDD Workshop 2006 | **Early Drift Detection Method** | — | Early Drift Detection, Error-Distance Monitoring, Gradual Drift, EDDM | [Paper](https://citeseerx.ist.psu.edu/document?doi=2747577a61c70bc3874380130615e15aff76339e&repid=rep1&type=pdf) |
| SDM 2007 | **Learning from Time-Changing Data with Adaptive Windowing** | — | Adaptive Windowing, Online Change Detection, Streaming Statistics | [Paper](https://doi.org/10.1137/1.9781611972771.42) |
| Discovery Science 2007 | **Detecting Concept Drift Using Statistical Testing** | — | Statistical Testing, Error-Rate Drift, Equal-Proportion Test, STEPD | [Paper](https://doi.org/10.1007/978-3-540-75488-6_27) |
| Pattern Recognition Letters 2012 | **Exponentially Weighted Moving Average Charts for Detecting Concept Drift** | — | EWMA Control Charts, Misclassification Monitoring, ECDD | [Paper](https://arxiv.org/abs/1212.6018) |
| ICDM 2014 | **Detecting Volatility Shift in Data Streams** | — | Volatility Shift, Block-Based Windowing, SEED, Drift Dynamics | [Paper](https://doi.org/10.1109/ICDM.2014.50) |
| IEEE TKDE 2015 | **Online and Non-Parametric Drift Detection Methods Based on Hoeffding’s Bounds** | — | Hoeffding Bounds, Non-Parametric Detection, HDDMA, HDDMW | [Paper](https://doi.org/10.1109/TKDE.2014.2345382) |
| Expert Systems with Applications 2017 | **RDDM: Reactive drift detection method** | — | Reactive Drift Detection, Recent Evidence, Error-Rate Monitoring, RDDM | [Paper](https://doi.org/10.1016/j.eswa.2017.08.023) |
| Machine Learning 2023 | **STUDD: A Student-Teacher Method for Unsupervised Concept Drift Detection** | Vitor Cerqueira | Unsupervised Drift Detection, Student–Teacher Disagreement, Label-Free Monitoring | [Paper](https://doi.org/10.1007/s10994-022-06188-7), [arXiv](https://arxiv.org/abs/2103.00903) |
| DMKD 2024 | **Adaptive Bernstein Change Detector for High-Dimensional Data Streams** | Marco Heyden | High-Dimensional Drift, Bernstein Bounds, Subspace Localization, ABCD | [Paper](https://doi.org/10.1007/s10618-023-00999-5), [arXiv](https://arxiv.org/abs/2306.12974), [Code](https://github.com/heymarco/AdaptiveBernsteinChangeDetector) |
| arXiv 2024 | **Sequential Change Point Detection in High-dimensional Vector Auto-regressive Models** | — | Sequential Change-Point Detection, High-Dimensional VAR, Transition-Matrix Shift | [Paper](https://arxiv.org/abs/2412.09794), [Code](https://github.com/Helloworld9293/VARcpDetectOnline) |

## Forecast and Adaptation

| Venue | Paper | Corresp. Author | Key Words | Links |
| --- | --- | --- | --- | --- |
| ICLR 2022 | **Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift** | — | Distribution Shift, Reversible Normalization, Instance Statistics, RevIN | [Paper](https://openreview.net/forum?id=cGDAkQo1C0p), [Code](https://github.com/ts-kim/RevIN) |
| Book Chapter 2023 | **Handling Concept Drift in Global Time Series Forecasting** | — | Global Forecasting, AR(3) Simulation, Sudden Drift, Gradual and Incremental Drift | [Paper](https://arxiv.org/abs/2304.01512), [Code](https://github.com/Neal-Liu-Ziyi/Concept_Drift_Handling), [Publication](https://doi.org/10.1007/978-3-031-35879-1_7) |
| AAAI 2023 | **Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting** | Pengyang Wang; Yanjie Fu | Intra-Space Shift, Inter-Space Shift, Dual Distribution Learning, Dish-TS | [Paper](https://arxiv.org/abs/2302.14829), [Code](https://github.com/weifantt/Dish-TS) |
| arXiv 2024 | **Addressing Concept Shift in Online Time Series Forecasting: Detect-then-Adapt** | — | Detect-then-Adapt, Loss-Distribution Monitoring, Delayed Feedback, Historical Data Augmentation | [Paper](https://arxiv.org/abs/2403.14949), [Code](https://github.com/yfzhang114/OneNet) |
| arXiv 2024 | **Online Test-Time Adaptation of Spatial-Temporal Traffic Flow Forecasting** | Lei Bai; Yu Zhang | Online Test-Time Adaptation, Temporal Drift, Seasonal–Trend Correction, ADCSD | [Paper](https://arxiv.org/abs/2401.04148), [Code](https://github.com/Pengxin-Guo/ADCSD) |
| KDD 2024 | **Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift** | Zhuo Li; Chenghao Liu | Context-Driven Shift, Residual-Based Detection, Contextualized Adaptation, SOLID | [Paper](https://arxiv.org/abs/2310.14838), [Publication](https://doi.org/10.1145/3637528.3671926) |
| SN Computer Science 2025 | **Dynamic Swarm Intelligence for Time Series Forecasting in the Presence of Concept Drift** | Gustavo H. F. M. Oliveira | Dynamic Swarm Intelligence, Drift Adaptation, Recurring Concepts, SISC | [Paper](https://doi.org/10.1007/s42979-025-04247-z), [Code](https://github.com/GustavoHFMO/SISC) |
| ICML 2025 Workshop (PUT) | **Shift-Aware Test-Time Adaptation and Benchmarking for Time-Series Forecasting** | — | Shift-Aware Test-Time Adaptation, Semi-Synthetic Benchmark, TTFBench, DynaTTA | [Paper](https://openreview.net/forum?id=a399SmgWGl), [Code](https://github.com/shivam-grover/DynaTTA) |
| arXiv 2026 | **When Tomorrow Becomes Today: Self-Evolving Policies for Agentic Time-Series Forecasting** | Zewei Dong; Qiang Xu | Policy Evolution, Delayed Feedback, Expert Trust, Adaptive Forecast Orchestration | [Paper](https://arxiv.org/abs/2609.24862) |

## Benchmark

| Venue | Paper | Corresp. Author | Key Words | Links |
| --- | --- | --- | --- | --- |
| DMKD 2020 | **Challenges in benchmarking stream learning algorithms with real-world data** | Vinicius M. A. Souza | Real-World Drift Benchmarking, Controlled Environmental Changes, Stream Classification, INSECTS | [Paper](https://doi.org/10.1007/s10618-020-00698-5), [arXiv](https://arxiv.org/abs/2005.00113) |
| PVLDB 2025 | **Stress-Testing ML Pipelines with Adversarial Data Corruption** | — | Structured Data Corruption, Adversarial Stress Testing, Constrained Search, SAVAGE | [Paper](https://www.vldb.org/pvldb/vol18/p4668-zhu.pdf), [arXiv](https://arxiv.org/abs/2506.01230), [Code](https://github.com/lodino/savage) |
| KDD 2026 | **A Framework for Evaluating and Benchmarking Concept Drift Detection Methods** | — | Drift Benchmarking, Semi-Synthetic Drift, Timing-Aware Evaluation, Hyperparameter Selection | [Paper](https://arxiv.org/abs/2606.07789), [Code](https://github.com/vcerqueira/experiments-drift_evaluation) |


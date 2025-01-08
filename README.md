# Introduction

Collaborative industrial project which uses Deep Reinforcemnt Learning Agent to efficiently allocate tasks in nodes in an adaptive distributed embedded system. Additionally the agent handles critical tasks ensuring fail-safety compliance and optimizes message passing among tasks, solving a NP Hard combinatorial problem in linear time.

# Installation

Make sure you have latest version of **Python 3.9** installed

```
pip install -r requirements.txt
```

# Inference

Navigate to the `src` folder and run:

**Format**

```
python main.py --config [PATH_CONFIG_1] [PATH_CONFIG_2] [--Param_Header1] [Param_Value1] .. so on
```

**Example**

`python main.py --train false --model_path ../experiments/models/p1/trnc_c/early_term_1000 --config utils/configs/problem_1.yaml utils/configs/experiment_trnc_c.yaml --experiment_name custom_experiments --run_name first_inference`

**Note:** Each and every parameter in existing configuration files is modifable. It can be changed and treated as a command line argument by putting double dash (--) as prefix.

# Training

`python main.py --config utils/configs/problem_1.yaml utils/configs/experiment_tn.yaml --experiment_name custom_experiments --run_name first_train`

**Note:** You may also provide your own custom configuration file

# Experiments

## Problem Sets and Configuration Variants

The study defines three problem sets and multiple configuration variants to evaluate the performance of RL agents in a CADES (Configurable Adaptive Distributed Execution System). These scenarios aim to emulate the dynamic and unpredictable conditions of real-world systems.

### Problem Sets

1. **Problem 1**: A static system configuration with fixed tasks and nodes. This serves as a baseline to evaluate basic performance.
2. **Problem 2**: Introduces variability in task numbers and costs while keeping nodes constant. It reflects fluctuating task demands with stable hardware resources.
3. **Problem 3**: Adds complexity by varying tasks, their costs, and the number of nodes. This scenario includes potential node downtimes, representing real-life challenges with dynamic task demands and resource failures.

| Problem No. | Tasks (#)    | Task Cost   | Nodes (#)    | Node Capacity |
|-------------|--------------|-------------|--------------|---------------|
| 1           | 12           | 4           | 6            | 12            |
| 2           | 8 to 10      | 4 to 6      | 6            | 12            |
| 3           | 8 to 10      | 4 to 6      | 6 to 8       | 10 to 12      |

### Configuration Variants

To capture different scenarios that may arise during the reconfiguration of a CADES, we propose several distinct configuration variants:

1. **TN**: Tasks and nodes are available, but no replicas or communication are required. Represents non-critical, independent task execution scenarios.
2. **TRN**: Adds replicas for critical tasks but no communication. Focuses on fault tolerance for critical tasks.
3. **TRNC**: Includes tasks, nodes, replicas, and communication, divided into:
   - **TRNC A**: Communication among non-critical tasks.
   - **TRNC B**: Communication among critical tasks.
   - **TRNC C**: Combines communication for both critical and non-critical tasks.

Each of these variants captures different levels of complexity, reflecting the diverse operational conditions that a CADES may encounter during reconfiguration.

| Category  | Tasks (T) | Nodes (N) | Replicas (R) | Communication (C)                  |
|-----------|-----------|-----------|--------------|-------------------------------------|
| TN        | ✔         | ✔         | ✘            | ✘                                   |
| TRN       | ✔         | ✔         | ✔            | ✘                                   |
| TRNC A    | ✔         | ✔         | ✔            | Non-critical tasks only            |
| TRNC B    | ✔         | ✔         | ✔            | Critical tasks only                |
| TRNC C    | ✔         | ✔         | ✔            | Both non-critical and critical tasks |

### Invalid Action Strategies

Different invalid action handling strategies are employed to conduct a comparative study of their effects on different configuration problems. These techniques are referenced in the results section and are summarized as follows:

1. **Early-Term**: This technique stands for **Early Termination** and applies termination for invalid actions.
2. **Act-Replace**: This technique stands for **Action Replacement** and applies replacement mechanism for invalid actions.
3. **Act-Mask**: This technique stands for **Action Masking** and applies logits masking for invalid actions.

These strategies are evaluated to understand their impact on solving different configuration problems effectively.

### Summary

The combination of problem sets and configuration variants provides a comprehensive framework for evaluating the RL agent's ability to handle dynamic, real-world challenges in a CADES. These scenarios test the agent's fault tolerance, adaptability, and task allocation efficiency under varying levels of complexity.

# Results

### Success Rate (%)

| Problem No. | Strategy      | TN   | TRN  | TRNC A | TRNC B | TRNC C |
|-------------|---------------|------|------|--------|--------|--------|
| **1**       | Early-Term    | 93   | 95   | 65     | **100**| **95** |
|             | Act-Replace   | **100** | **98** | **87** | 93     | 88     |
|             | Act-Mask      | **100** | 97   | 69     | 52     | 49     |
| **2**       | Early-Term    | **100** | 98   | **98** | **99** | 96     |
|             | Act-Replace   | 98   | **99** | **98** | **99** | **98** |
|             | Act-Mask      | 98   | 98   | 93     | 96     | 96     |
| **3**       | Early-Term    | 94   | **97** | 84     | **88** | 89     |
|             | Act-Replace   | **97** | 95   | 84     | **88** | **90** |
|             | Act-Mask      | 95   | 96   | 84     | 85     | 79     |

**Note**: Bolded values indicate the highest performance in each category.

Detailed results can be found in the paper.

# Future Work

Optimize deep learning agent to fulfill message passing among tasks more efficiently
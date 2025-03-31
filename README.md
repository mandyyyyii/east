# EAST: Entropy-Based Adaptive Weighting for Self-Training
**EAST** is an adaptive weighting strategy designed to prioritize uncertain data during self-training. Specifically, EAST employs a mapping function with a tunable parameter that controls the sharpness of the weighting, assigning higher weights to data where the model exhibits greater uncertainty. This approach guides the model to focus on more informative and challenging examples, thereby enhancing its reasoning ability.

<!-- ![EAST Comparison](assets/east_compare.png)
*Figure 1: Comparison between the traditional self-training pipeline and EAST. The LLM generates $n$ responses per question, clustered by final answers. Questions with all incorrect answers are discarded. Self-training fine-tunes uniformly on the rest, while EAST assigns higher weights to questions with diverse (uncertain) answers and lower weights to consistent (confident) ones.* -->

![EAST Overview](assets/east_overview.png)
*Figure 1: The framework of EAST. For each training question, the LLM generates $n$ responses, clustered by final answers. Entropy value is computed from the cluster distribution, transformed via mapping function, and integrated as weight into the loss objective.*

## Updates
[Mar 2025] We release the code and arxiv the paper. 

## Training
Our code is built upon [trl repository](https://github.com/huggingface/trl/tree/main). 

### Training Scripts
```
bash train/run_east_dpo.sh
```
## Results

<table>
  <thead>
    <tr>
      <th rowspan="2">Setting</th>
      <th colspan="3" style="text-align:center">LLaMA-3.2-1B</th>
      <th colspan="3" style="text-align:center">LLaMA-3.1-8B</th>
    </tr>
    <tr>
      <th>GSM8K (%)</th>
      <th>MATH (%)</th>
      <th>AVG (%)</th>
      <th>GSM8K (%)</th>
      <th>MATH (%)</th>
      <th>AVG (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><i>default</i></td><td>46.2</td><td>28.5</td><td>37.3</td><td>82.8</td><td>50.4</td><td>66.6</td></tr>
    <tr><td>SFT</td><td>50.1</td><td>28.4</td><td>39.2</td><td>85.0</td><td>50.0</td><td>67.5</td></tr>
    <tr><td>+EAST</td><td>51.8</td><td>29.4</td><td>40.6</td><td>86.1</td><td>51.2</td><td>68.6</td></tr>
    <tr><td>DPO</td><td>50.2</td><td>28.7</td><td>39.5</td><td>84.6</td><td>50.1</td><td>67.5</td></tr>
    <tr><td>+EAST</td><td>51.9</td><td>29.7</td><td>40.8</td><td>85.4</td><td>50.9</td><td>68.1</td></tr>
    <tr><td>KTO</td><td>53.0</td><td>28.8</td><td>40.9</td><td>83.9</td><td>48.9</td><td>66.4</td></tr>
    <tr><td>+EAST</td><td>53.0</td><td>29.9</td><td>41.5</td><td>85.1</td><td>51.0</td><td>68.1</td></tr>
  </tbody>
</table>



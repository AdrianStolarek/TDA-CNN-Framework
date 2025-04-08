
# TDA-CNN Framework

This project investigates whether combining raw images with vectorized persistence diagrams (persistence images) can enhance CNN classification accuracy. We evaluate this approach across both grayscale and RGB image classification tasks using various CNN architectures. Ready-to-use functions for Optuna-based hyperparameter optimization are included. 

### Research Hypothesis

We test the hypothesis that feature concatenation of original image data with topological features extracted through persistent homology can improve model performance compared to using either data type alone.

I'm curious how this project may evolve. Contributions to expand this research are very much welcome! Feel free to suggest improvements, new experiments, or additional architectures to test :)


We measure the classification performance of a CNN (default: ResNet50) trained on:
- Original unprocessed datasets
- Datasets enhanced with topological features
- Pure persistence images alone

Performance is evaluated using multiple metrics (accuracy, precision, recall, F1-score).

We investigate data efficiency by:
- Determining the minimum dataset size required when using topological enhancement
- Evaluating various dataset reduction strategies (random sampling, confidence-based filtering, topological significance ranking)
- Documenting how reduction impact varies with dataset complexity

This repository includes a detailed research report containing:
- Comprehensive benchmark results with methodological explanations
- Discusses key findings and their implications for TDA-enhanced CNN training
- Clear documentation of experimental assumptions
- Identifies potential optimization opportunities and research questions

Sample results below (read the report for more context):

  ![final_long_1000](https://github.com/user-attachments/assets/cc86a095-287d-4854-b94f-5059d4969a53)

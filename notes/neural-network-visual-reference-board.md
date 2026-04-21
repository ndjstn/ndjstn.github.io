# Neural Network Visual Reference Board

This is the inspiration board for improving the article graphics without copying anyone else's layouts or art.

## What the current graphics were missing

- Too much decorative shape language, not enough measurement.
- Not enough plain labeling of what changes, what stays fixed, and why that matters.
- Weak encoding of sign and magnitude for weights.
- Bias shown as a shifted curve, but not connected clearly to threshold movement or decision-boundary translation.
- Hidden-layer graphic showed points moving, but not why the representation became easier to classify.
- Backprop graphic showed a process flow, but not enough causal structure.

## Design rules for the next pass

- Show one concrete numerical example whenever possible.
- Put scales on anything quantitative.
- Use color for sign and intensity for magnitude.
- Carry the same symbols from prose to figure: `x`, `w`, `b`, `z`, `\hat{y}`.
- Prefer “before / after / after” small multiples over one abstract cartoon.
- Use animation only when motion explains the concept, not as decoration.

## Reference examples

### Problem framing and system overview

1. Google ML Crash Course — Neural Networks  
   `https://developers.google.com/machine-learning/crash-course/neural-networks`
2. Google ML Crash Course — Nodes and Hidden Layers  
   `https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers`
3. TensorFlow Playground  
   `https://playground.tensorflow.org/`
4. CS231n — Neural Networks Part 1  
   `https://cs231n.github.io/neural-networks-1/`
5. 3Blue1Brown — Gradient Descent, How Neural Networks Learn  
   `https://www.3blue1brown.com/lessons/gradient-descent`

### Neurons, weights, and bias

6. Google ML Crash Course — Nodes and Hidden Layers  
   `https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers`
7. TensorFlow Playground  
   `https://playground.tensorflow.org/`
8. CS231n — Neural Networks Part 1  
   `https://cs231n.github.io/neural-networks-1/`
9. 3Blue1Brown — What Is Backpropagation Really Doing?  
   `https://www.3blue1brown.com/lessons/backpropagation`
10. Google ML Crash Course — Thresholds and the Confusion Matrix  
    `https://developers.google.com/machine-learning/crash-course/classification/thresholding`

### Activation functions and threshold behavior

11. Google ML Crash Course — Activation Functions  
    `https://developers.google.com/machine-learning/crash-course/neural-networks/activation-functions`
12. Google ML Crash Course — Sigmoid Function  
    `https://developers.google.com/machine-learning/crash-course/logistic-regression/sigmoid-function`
13. Google ML Crash Course — Thresholds and the Confusion Matrix  
    `https://developers.google.com/machine-learning/crash-course/classification/thresholding`
14. TensorFlow Playground  
    `https://playground.tensorflow.org/`
15. CS231n — Neural Networks Part 1  
    `https://cs231n.github.io/neural-networks-1/`

### Hidden layers and representation learning

16. Google ML Crash Course — Neural Networks  
    `https://developers.google.com/machine-learning/crash-course/neural-networks`
17. Google ML Crash Course — Nodes and Hidden Layers  
    `https://developers.google.com/machine-learning/crash-course/neural-networks/nodes-hidden-layers`
18. scikit-learn — `DecisionBoundaryDisplay`  
    `https://scikit-learn.org/stable/modules/generated/sklearn.inspection.DecisionBoundaryDisplay.html`
19. scikit-learn — Plot Classification Probability  
    `https://scikit-learn.org/stable/auto_examples/classification/plot_classification_probability.html`
20. scikit-learn — Neural Networks Example Gallery  
    `https://scikit-learn.org/stable/auto_examples/neural_networks/index.html`
21. scikit-learn — Varying Regularization in Multi-layer Perceptron  
    `https://scikit-learn.org/stable/auto_examples/neural_networks/plot_mlp_alpha.html`
22. Distill — Feature Visualization  
    `https://distill.pub/2017/feature-visualization/`
23. Distill — Activation Atlas  
    `https://distill.pub/2019/activation-atlas/`

### Backpropagation and learning dynamics

24. Google ML Crash Course — Training Using Backpropagation  
    `https://developers.google.com/machine-learning/crash-course/neural-networks/backpropagation`
25. CS231n — Optimization Part 2: Backpropagation  
    `https://cs231n.github.io/optimization-2/`
26. 3Blue1Brown — What Is Backpropagation Really Doing?  
    `https://www.3blue1brown.com/lessons/backpropagation`
27. 3Blue1Brown — Backpropagation Calculus  
    `https://www.3blue1brown.com/lessons/backpropagation-calculus`
28. Michael Nielsen — How the Backpropagation Algorithm Works  
    `https://michaelnielsen.org/blog/how-the-backpropagation-algorithm-works/`

### Plotting patterns worth borrowing

29. Matplotlib — Animation Gallery  
    `https://matplotlib.org/stable/gallery/animation/index.html`
30. Matplotlib — Animated Scatter Saved as GIF  
    `https://matplotlib.org/stable/gallery/animation/simple_scatter.html`
31. Matplotlib — Slider Demo  
    `https://matplotlib.org/stable/gallery/widgets/slider_demo.html`
32. Matplotlib — Widgets Gallery  
    `https://matplotlib.org/stable/gallery/widgets/index.html`
33. Matplotlib — Annotated Heatmap  
    `https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html`
34. Matplotlib — Text, Labels, and Annotations  
    `https://matplotlib.org/stable/gallery/text_labels_and_annotations/index.html`
35. Matplotlib — RangeSlider  
    `https://matplotlib.org/stable/gallery/widgets/range_slider.html`

## What to borrow, not copy

- From TensorFlow Playground: sign by color, magnitude by line strength, and direct linkage between parameter changes and classification regions.
- From Google ML Crash Course: interactive, step-by-step “what changed?” explanations with plain labels and visible calculations.
- From CS231n: simple network diagrams plus direct ties between formulas and visuals.
- From Distill: small multiples, annotated transitions, and visuals that make internal representations feel concrete.
- From scikit-learn: probability surfaces and decision boundaries that show what a classifier is actually doing.
- From Matplotlib examples: sliders, animated sweeps, annotated heatmaps, and clearer callouts.

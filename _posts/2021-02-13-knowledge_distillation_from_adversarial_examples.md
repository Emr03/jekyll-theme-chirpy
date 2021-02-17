---
title: The Structure of Adversarial Perturbations
author: Elsa Riachi
categories: [Research]
tags: [adversarial robustness]
date: 2021-02-13
math: true
katex: true
---

<!-- Load KaTeX -->
<!-- <link rel="stylesheet" href="/assets/katex/katex.min.css">
<script src="/assets/katex/katex.min.js"></script>
<script defer src="/assets/katex/contrib/auto-render.min.js" onload="renderMathInElement(document.body);"></script> -->

## Introduction and Background
Adversarial vulnerability is a fundamental limitation of deep neural networks which remains poorly understood. Recent work suggests that adversarial attacks exploit the fact that non-robust models rely on superficial statistics to form predictions. Ilyas et al. discuss this hypothesis in [this article]() and [this paper](). In a nutshell, the authors propose that there may be patterns of pixels that are highly predictive of the image's class label but appear non-sensical to humans.

This hypothesis is reminiscent of results that show that neural networks are highly sensitive to changes in texture, or high-frequency components. Most importantly, the authors hypothesize that these patterns or *features* which appear non-sensical, allow the network to generalize to unseen examples. To support their claim, the authors demonstrate a surprising experimental result. Their method is briefly outlined below:

1. The authors train a neural network (let's call it network A) on the CIFAR10 training set.  
2. For each image-label pair (x, y) in the training set, a target label $$t$$ is chosen at random, and an adversarial perturbation
$$\delta$$ is computed such that network A classifies $$x + \delta$$ as an instance of class t.
3. The above procedure results in a newly created training set of adversarial input-target pairs $$(x + \delta, t)$$ denoted as $$D_{adv}$$.
4. A new network (let's call it netowrk B) is trained on the adversarial training set $$D_{adv}$$.
5. Network B is then evaluated on the original CIFAR10 test set, and surprisingly it does well!

The fact that network B, trained on $$D_{adv}$$ generalizes to the standard test set appears to contradict the basic intuition that generalization is achieved by training on many representative samples from the data distribution. The authors use this observation to conclude that adversarial perturbations introduce well-generalizing features that are predictive of the target label $$t$$.

In this article, I briefly outline my research which sheds light on the structure of adversarial perturbations and why generalization may be achieved from a training set of adversarial image-label pairs.

## What About Attacks on Autoencoders?
A natural approach to gain insight into the phenomenon described above is to replicate it using autoencoders. The intuition behind this approach is simple. Since an autoencoder is required to effectively compress an input image so that it may be reconstructed with low reconstruction error, an encoder must capture most of the discernible features within an image. Yet, autoencoders have also been shown to be vulnerable to adversarial attack. If an input image may be reconstructed from the encoder's representation, what makes the encoder vulnerable to adversarial attacks? How should we interpret the results of Ilyas et al. \cite{}?

% Results of sparse coding adversarial perturbations

% Why do classifiers generalize when trained on adversarial data? Special case: multi-class perceptron.

References
----------

{% bibliography --file adversarial_examples %}

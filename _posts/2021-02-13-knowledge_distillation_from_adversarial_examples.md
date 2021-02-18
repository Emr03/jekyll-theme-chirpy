---
title: The Structure of Adversarial Perturbations (Part I)
author: Elsa Riachi
categories: [Research]
tags: [adversarial robustness]
date: 2021-02-13
math: true
---

<div style="display:none">
Adversarial vulnerability is a fundamental limitation of deep neural networks which remains poorly understood. Recent work suggests that adversarial attacks exploit the fact that non-robust models rely on superficial statistics to form predictions.
</div>

<div style="display:none">
$$
\newcommand\testmacro[2]{\mathbf{F\alpha}(#1)^{#2}}
\def\norm#1{\left\|{#1}\right\|} % A norm with 1 argument
\newcommand\zeronorm[1]{\norm{#1}_0} % L0 norm
\newcommand\onenorm[1]{\norm{#1}_1} % L1 norm
\newcommand\twonorm[1]{\norm{#1}_2} % L2 norm
$$
</div>

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

Targeted attacks on autoencoders can be formulated as the constrained optimization problem shown below:

$$ \begin{align}
\boldsymbol{\delta}^* &= \mathop{\mathrm{arg} min} \twonorm{E(\textbf{x}_t) - E(\textbf{x}_s + \boldsymbol{\delta})}^2 \\
\label{eq:encoder_attack} \\
& \twonorm{\boldsymbol{\delta}}^2 \leq \epsilon
\end{align}$$

Where a norm-bounded perturbation $$\boldsymbol{\delta}$$ is added to a source image $$\textbf{x}_s$$ so as to produce a similar representation to that of a randomly selected target image $$\textbf{x}_t$$.  Denoting the encoder by $$E(.)$$ and the decoder by $$D(.)$$, the success of the attack is determined by the squared error between the target $$\textbf{x}_t$$ and the reconstruction $$D \circ E(\textbf{x}_s + \delta)$$. Examples of targeted adversarial attacks on images from the CelebA dataset are shown below.

<img src="images/attack_pair.png" width="750">

With the attack objective shown above, we generate a training set of adversarial input-target pairs $$(\textbf{x}_s + \boldsymbol{\delta}, \textbf{x}_t)$$, much like the experimental procedure described by Ilyas et al. \cite{}. Interestingly, we observe that a newly initialized autoencoder trained on the adversarial training set learns to reconstruct unperturbed images from the standard test set. In our case, the added perturbation $$\boldsymbol{\delta}$$ isn't merely predictive of a class label, but of the target image!

In the next section we examine the worst-case noise of a linear encoder to understand the structure of adversarial perturbations and obtain some insight into this strange observation.

## Adversarial Attacks on a Linear Encoder

In this section we study attacks on a linear encoder, represented as a matrix $$\Phi \in \mathbb{R}^{M \times N}$$, where $$M << N$$. While principal component analysis first comes to mind when constructing a linear encoder, we defer this discussion for later. For now, we focus on the worst-case perturbation $$\boldsymbol{\delta}$$ for a particular source-target pair $$(\textbf{x}_s, \textbf{x}_t)$$.

$$
\begin{align}
\min_{\delta} \twonorm{\Phi (\mathbf{x_s} + \boldsymbol{\delta}) - \Phi \mathbf{x_t}}^2 \\
\twonorm{\boldsymbol{\delta}}^2 \leq \epsilon^2
\label{eq:attack}
\end{align}
$$

Since the above constrained optimization problem is convex, the solution is the critical point of the Lagrangian $$\mathcal{L}(\boldsymbol{\delta}, \lambda)$$.

$$
\begin{align}
\mathcal{L}(\boldsymbol{\delta}, \lambda) &= (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta})^T \Phi^T \Phi (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta}) + \lambda(\boldsymbol{\delta}^T \boldsymbol{\delta} - \epsilon) \label{eq:lagrangian} \\
\nabla_{\delta}\mathcal{L}(\boldsymbol{\delta}, \lambda) &= 2\Phi^T \Phi (\textbf{x}_s - \textbf{x}_t + \boldsymbol{\delta}) + 2\lambda \boldsymbol{\delta} = 0 \nonumber \\
\boldsymbol{\delta} &= \left (\Phi^T \Phi + \lambda I \right)^{-1} \Phi^T \Phi \left (\textbf{x}_t - \textbf{x}_s \right) \nonumber \\
\end{align}
$$

The solution $$\boldsymbol{\delta}$$ can be decomposed into two components $$\boldsymbol{\delta_s}$$ and $$\boldsymbol{\delta_t}$$, where $$\boldsymbol{\delta_s}$$ is such that $$\twonorm{\Phi \left( \mathbf{x_s} - \boldsymbol{\delta_s}\right )}^2$$ is minimized, while $$\boldsymbol{\delta_t}$$ is such that $$\twonorm{\Phi \left( \boldsymbol{\delta_t} - \mathbf{x_t} \right)}^2$$ is minimized. That is, $$\boldsymbol{\delta_s}$$ is crafted so as to obfuscate $$\textbf{x}_s$$ while $$\boldsymbol{\delta_t}$$ is crafted so as to pass as $$\textbf{x}_t$$.  

$$
\begin{align}
\boldsymbol{\delta}_s &= \left (\Phi^T \Phi +  \lambda I \right)^{-1} \Phi^T \Phi \textbf{x}_s \label{eq:delta_s} \\
\boldsymbol{\delta}_t &= \left (\Phi^T \Phi +\lambda I \right)^{-1} \Phi^T \Phi \textbf{x}_t \label{eq:delta_t}
\end{align}
$$

We denote the transformation $$\left (\Phi^T \Phi + \lambda I \right)^{-1} \Phi^T \Phi $$ by the matrix $$\textbf{M}_{\Phi}$$.  The final expression for $$\boldsymbol{\delta}$$ which we use to attack $$\Phi$$ is shown below.


$$
\begin{equation}
\boldsymbol{\delta} = \textbf{M}_\Phi \textbf{x}_t - \textbf{M}_\Phi \textbf{x}_s
\label{eq:final_delta}
\end{equation}
$$


We now consider the case where the linear encoder $$\Phi$$ is constructed from the top $$M$$ principal components of the input distribution. That is, the rows of $$\Phi$$ are orthonormal vectors in $$\mathbb{R}^{N}$$. An input vector $$\textbf{x}$$ consists of a linear combination of the top $$M$$ principal components $$\{\boldsymbol{\phi_1}, ...\boldsymbol{\phi_M}\}$$, which we denote by $$\hat{\textbf{x}}$$, and a component that is orthogonal to the span of $$\{\boldsymbol{\phi_1}, ...\boldsymbol{\phi_M}\}$$. We denote the orthogonal component as $$\textbf{n}$$. Note that $$\twonorm{\textbf{n}} = \twonorm{\hat{\textbf{x}} - \textbf{x}}$$. Since the principal components are such that $$\twonorm{\hat{\textbf{x}} - \textbf{x}}$$ is minimized, we may consider the reconstruction error $$\twonorm{\textbf{n}}$$ to be small.


$$
\begin{equation}
\textbf{x} = \sum_{i=1}^{M} \alpha_i \boldsymbol{\phi_i} + \textbf{n}
\end{equation}
$$


The output of $$\Phi \textbf{x}$$ is $$(\alpha_1, \alpha_2, ..., \alpha_MS)^T$$ whose norm must be much larger than $$\twonorm{\textbf{n}}$$ if the number of principal components is chosen well. We can immediately see that for a perturbation $$\boldsymbol{\delta}$$ to be successful, we 



% Discuss delta in terms of singular vectors
% infomax with non-invertible transformation
% input must live in a lower dimensional space
%

We begin by studying a motivating toy example using a synthetic dataset of structured sparse signals. Our constructed dataset consists of $28 \times 28$ images made up of at most 5 discrete Fourier transform (DFT) components. A $28 \times 28$ image can consist of at most $(28 / 2 + 1)^2$ = 225 discrete Fourier frequencies \citep{Bracewell1965TheFT}; however, to make our synthetic images easily discernible, we restrict our dataset to only contain 28 frequencies corresponding to periodic signals along either the horizontal or vertical axes of the image, but not both. Furthermore, only five frequencies may be present in a single image, the combination of which is selected from a set of 200 possible configurations. We denote the sparse representation in the DFT domain of a $28 \times 28$-dimensional image as

% Why do classifiers generalize when trained on adversarial data? Special case: multi-class perceptron.

References
----------

{% bibliography --file adversarial_examples %}

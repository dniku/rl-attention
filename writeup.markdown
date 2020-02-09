# Regularization and visualization of attention in reinforcement learning agents

Dmitry Nikulin, Sebastian Kosch, Fabian Steuer, Hoagy Cunningham

<p><a href="https://github.com/dniku/rl-attention">Github Repository</a></p>

*This project was completed during AI Safety Camp 3 in Ávila, Spain, in May
2019. The goal was to familiarize ourselves with the state of the art in
visualizing machine learning models, particularly in an reinforcement learning
context, and to test possible improvements.*

## Introduction

Advances in deep learning are enabling reinforcement learning (RL) agents to
accomplish increasingly difficult tasks. For instance, relatively simple machine
learning agents can learn how to beat humans in video games, without ever having
been programmed how to do so. However, agents sometimes learn to make correct
decisions for the wrong reasons, which can lead to surprising and perplexing
failures later. In order to diagnose such problems effectively, the developer
needs to understand how information flows through the artificial neural network
that powers the agent's decision-making process.

One approach to understanding complex models operating on image inputs is
through saliency maps. A saliency map is a heatmap highlighting those pixels of
the input image that are most responsible for the model output. An example could
be a neural network performing an image classification task: given a photograph
of a dog in a meadow that was correctly classfied as "dog", a visualization of
those pixels the network considers most dog-like provides a check on whether the
network has truly learned the concept of dogs, or whether it merely made a lucky
guess based on the presence of the meadow.

To some extent, the methods for generating saliency maps can be repurposed for
the analysis of RL agents playing video games, since the RL infers actions
(labels) from images (render frames). However, the resulting heatmaps are often
blurry or noisy. Furthermore, image classifiers simply detect objects, while RL
agents must choose actions based on complex relationships between entities
detected in the input. A simple heatmap visualization cannot convey whether and
where such relationships were detected.

In this work, we present two potential improvements to existing visualization
approaches in RL, and report on our experimental findings regarding their
performance on the Atari Breakout game.

## Attention in RL agents

In 2016, [Yang et al.](https://arxiv.org/abs/1812.11276) explored the effects of
adding two *attention layers* to the decision-making network of an agent
learning to play Breakout and other games. The attention layers, applied after
some convolutional layers which detect basic game entities, restrict the input
to the agent's action selection mechanism to a subset of the input pixels (the
*attention mask*). In effect, the agent's model is forced to focus spatially.
Adding such a bottleneck can improve sample efficiency, but more importantly,
the attention layer activations provide a direct clue about what the model is
focusing on. This directness is attractive when compared to *post-hoc* methods,
which require additional computation to reason about the relevance of network
activations after inference.

<p><img src="images/architecture.png" class="archi" alt="Diagram of the architecture used in our models." width="600"/></p>

Note that there is no direct correspondence between activated attention layer
neurons and relevant input pixels. This is due to the convolutional downsampling
layers that separate the input image from the attention layers. However, we can
generate a heatmap by backpropagating the attention tensor through the network.
Several different approaches exist to accomplish this, from Simonyan et al.'s
gradient method to the more recent VarGrad and SmoothGrad sampling methods.

After some experimentation, Yang et al. chose a simpler approach, where they
simply visualized the receptive field of the neuron which corresponded to the
strongest activation in their agent's attention layer. Their findings confirm
that in trained agents, the attention tends to be strong near crucial entities
in the game, i.e. the Pacman sprite or the moving ball in Breakout. However, the
attention mask heatmaps are fairly crude.

## Adding regularization to sharpen the attention masks

The effectiveness of attention layers depends crucially on how the attention is
constrained. This is especially true because of the downsampling action of the
convolutional layers in Yang et al.'s architecture: a diffuse attention tensor
will effectively correspond to all input pixels, defeating the purpose of the
attention layer.

To incentivize more informative heatmaps than those obtained by Yang et al., we
added an extra loss term to represent the diffuseness of the attention tensor.
Several such measures exist; we settled on using the entropy of the final
attention layer.

For a discrete probability distribution \\(p\_i, i=1..n\\) entropy is defined as
follows:

\\\[ \\operatorname{entropy}(p) = -\\sum\_{i = 1}^n p\_i \\cdot \\log(p\_i).
\\\]

This quantity is greatest when \\(p\_i \\equiv \\frac 1 n\\) for all \\(i\\),
i.e., when the corresponding probability distribution is very evenly spread. By
contrast, it is equal to zero when \\(p\_i = 1\\) for some \\(i\\), while the
other outcomes have probability zero.

In our case, we regard the output of the attention layer as a probability
distribution, and we modify the loss in such way as to minimize the entropy of
this distribution. Specifically, we add \\(\\lambda \\cdot
\\operatorname{entropy}(attn)\\) to the loss, where \\(\\lambda\\) is a
non-negative coefficient and \\(attn\\) is the output of the attention layer.

Although attention mechanisms have been shown to improve training times,
excessively strong regularization will naturally prevent the agent from taking
into account complex relationships between spatially distant entities, and thus
degrade performance. We ran a suite of experiments to quantify the impact of
entropy regularization on the agent's performance at playing Breakout.

### Experimental Results

We recreated the agent by Yang et al. in TensorFlow using the `stable-baselines`
package, a fork of OpenAI's `baselines` package with a stable programming
interface and improved documentation. Since `stable-baselines` does not include
a full implementation of the Rainbow algorithm, we used PPO as another
state-of-the-art algorithm available in the library.

Our experiments show that entropy regularization can be added in such way that
its effects on attention maps become pronounced, but performance does not suffer
much. The following figure shows average reward agents obtain during training
with varying coefficient before entropy loss. `0` means that entropy loss did
not affect training and should be regarded as baseline.

<div class="figure">
<img src="images/reward_curves.png" alt="Average reward during training" class="center"/>
<p class="caption">Average reward during training</p>
</div>

From the data it is clear that \\(\\lambda\\) equal to 0.0005 does not lead to any
drop in performance. The following figure shows impact on the resulting entropy
value.

<div class="figure">
<img src="images/scatterplots.png" alt="Scatterplot of final performance. X-axis: attention entropy. Y-axis: average reward." class="center"/>
<p class="caption">Scatterplot of final performance. X-axis: attention entropy. Y-axis: average reward. Solid circles denote individual runs with various random seeds. Cross marks denote averages across runs.</p>
</div>

With the exception for BeamRider, it is clear that for the particular case of
these Atari games, it is possible to choose a value of \\(\\lambda\\) such that
the extra term in the loss will have a noticeable effect on entropy value, while
final performance will not suffer.

BeamRider is different because for these particular training runs, none of the
agents achieved good performance. This is expected: in the original PPO paper
([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) it is reported that learning only starts after roughly 10M frames (we terminate training after exactly 10M
frames).

The following videos show what learned attention maps look like with different \\(\\lambda\\).

<figure class="video_container">
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/BeamRider.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/Breakout.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/MsPacman.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/Frostbite.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/Enduro.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="700", align="center">
    <source src="videos/Seaquest.mp4" type="video/webm">
</video>
</figure>

## Augmenting visualization of attention using tree structures

Heatmaps are an excellent tool to visualize the relevance of individual pixels
to the agent's decision. But asking about the relevance of individual pixels
rarely results in satisfying explanations.

<p align="center">
<b> Simonyan Gradient (Left), Smoothgrad (Right) and Vargrad (Bottom) of the attention layer without entropy penalty. </b>
</p>
<figure class="video_container" , align="center">
<video controls="true" allowfullscreen="true", width="350", align="left">
    <source src="images/simonyan_no_attn.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="350", align="right">
    <source src="images/coef_0.0_sum_smoothgrad_50.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="350", align="center">
    <source src="images/coef_0.0_sum_vargrad_50.mp4" type="video/webm"> </video>
</figure>

Visualizations of the filters learned by the convolutional layers of the network
suggest that lower layers detect the presence (or absence) of particular game
entities, such as a paddle or a ball, while higher layers encode spatial
relationships between those entities. Therefore, the presence of certain spatial
relationships between game entities informs the decision-making.

We propose that two-dimensional heatmaps can be replaced with interactive,
three-dimensional visualizations that present such spatial relationships in a
manner analogous to Bach et al.'s layer-wise relevance propagation (LRP)
approach. Beginning with the attention layer, we select the *n* strongest
activations, and then recursively find the *n* strongest contribution
activations in the layers below. This results in a tree structure, a
visualization of which can show not only where the presence (or absence) of
specific game entities is most relevant, but also which spatial relationships
between them most contributed to the selected action.

In practice, it can be challenging to select the most important elements in each
layer based on their activations, since neighbouring neurons in convolutional
layers share very similar inputs, and we are interested in prominent peaks, not
absolute peaks, in activation strength. Our initial choice of k-means clustering
to accomplish this proved too slow for interactive visualizations, so we
switched to a peak detection algorithm to find neurons of interest. The final
tree is plotted on top of the stack of input frames.

<div class="footnotes">
<hr />
<ol>
<li id="fn1"><p>Zhao Yang et al. <a href="https://arxiv.org/pdf/1812.11276.pdf">Learn to Interpret Atari Agents</a><a href="#fnref1">↩</a></p></li>
<li id="fn2"><p>Simonyan et al. <a href="https://arxiv.org/abs/1312.6034.pdf">Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps</a><a href="#fnref2">↩</a></p></li>
</ol>
</div>

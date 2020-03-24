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

In this work, we present a potential improvement to existing visualization
approaches in RL, and report on our experimental findings regarding their
performance on six Atari games.

## Attention in RL agents

In 2018, Yang et al.<sup id="fnref1">[\[1\]](#fn1)</sup> explored the effects of adding two *attention
layers* to the decision-making network of an agent learning to play Breakout and
other games. The attention layers, applied after some convolutional layers which
detect basic game entities, restrict the input to the agent's action selection
mechanism to a subset of the input pixels (the *attention mask*). In effect, the
agent's model is forced to focus spatially. Adding such a bottleneck can improve
sample efficiency, but more importantly, the attention layer activations provide
a direct clue about what the model is focusing on. This directness is attractive
when compared to *post-hoc* methods, which require additional computation to
reason about the relevance of network activations after inference.

Note that there is no direct correspondence between activated attention layer
neurons and relevant input pixels. This is due to the convolutional downsampling
layers that separate the input image from the attention layers. However, we can
generate a heatmap by backpropagating the attention tensor through the network.
Several different approaches exist to accomplish this, from the gradient method introduced by Simonyan et al.<sup id="fnref2">[\[2\]](#fn2)</sup> to the more recent SmoothGrad<sup id="fnref5">[\[5\]](#fn5)</sup> and VarGrad<sup id="fnref6">[\[6\]](#fn6)</sup> sampling
methods.

After some experimentation, Yang et al. chose a simpler approach, where
they simply visualized the receptive field of the neuron which corresponded to
the strongest activation in their agent's attention layer. Their findings
confirm that in trained agents, the attention tends to be strong near crucial
entities in the game, i.e. the Pacman sprite or the moving ball in Breakout.
However, the attention mask heatmaps are fairly crude.

## Adding regularization to sharpen the attention masks

The effectiveness of attention layers depends crucially on how the attention is
constrained. This is especially true because of the downsampling action of the
convolutional layers in Yang et al.'s architecture: a diffuse attention
tensor will effectively correspond to all input pixels, defeating the purpose of
the attention layer.

To incentivize more informative heatmaps than those obtained by Yang et al., we added an extra loss term to represent the diffuseness of the
attention tensor. Several such measures exist; we settled on using the entropy
of the final attention layer.

<div class="figure">
<img src="images/architecture.png" class="center" alt="Diagram of the architecture used in our models." width="600"/>
<p class="caption">Figure 1: The architecture of Yang et al., which we used in all experiments. The place where we applied entropy loss is highlighted.</p>
</div>

For a discrete probability distribution \\(p\_i, i=1..n\\) entropy is defined as
follows:

\\\[ \\operatorname{entropy}(p) = -\\sum\_{i = 1}^n p\_i \\cdot \\log(p\_i).
\\\]

This quantity is greatest when \\(p\_i \\equiv \\frac 1 n\\) for all \\(i\\),
i.e., when the corresponding probability distribution is very evenly spread. By
contrast, it is equal to zero when \\(p\_i = 1\\) for some \\(i\\), while the
other outcomes have zero probability.

In our case, we regard the output of the attention layer as a probability
distribution, and we modify the loss in such way as to minimize the entropy of
this distribution. Specifically, we add \\(\\lambda \\cdot
\\operatorname{entropy}(attn)\\) to the loss, where \\(\\lambda\\) is a
non-negative coefficient and \\(attn\\) is the output of the attention layer.

Although attention mechanisms have been shown to improve training times,
excessively strong regularization will naturally prevent the agent from taking
into account complex relationships between spatially distant entities, and thus
degrade performance. We ran a suite of experiments to quantify the impact of
entropy regularization on the agent's performance at playing Atari games.

### Experimental Results

We recreated the agent by Yang et al. in TensorFlow using the
`stable-baselines`<sup id="fnref3">[\[3\]](#fn3)</sup> package, a fork of OpenAI's `baselines` package with
a stable programming interface and improved documentation. Since
`stable-baselines` does not include a full implementation of the Rainbow
algorithm, we used PPO<sup id="fnref4">[\[4\]](#fn4)</sup> as another
state-of-the-art algorithm available in the library.

Our experiments show that entropy regularization can be added in such way that
its effects on attention maps become pronounced, but performance does not suffer
noticeably. The following figure shows average reward agents obtain during
training with varying \\(\\lambda\\). The value \\(\\lambda = 0.0\\) means that
entropy loss did not affect training. This setup should be regarded as baseline.

<div class="figure">
<img src="images/reward_curves.png" alt="Average reward during training" class="center"/>
<p class="caption">Figure 2: Average reward during training.</p>
</div>

From the data it is clear that \\(\\lambda = 0.0005\\) does not lead to any drop
in performance. The following figure shows impact on the resulting entropy
value.

<div class="figure">
<img src="images/scatterplots.png" alt="Figure 2: Scatterplot of final performance. X-axis: attention entropy. Y-axis: average reward." class="center"/>
<p class="caption">Figure 3: Scatterplot of final performance. X-axis: attention entropy. Y-axis: average reward. Solid circles denote individual runs with various random seeds. Cross marks denote averages across runs.</p>
</div>

With the exception for BeamRider, it is clear that for the particular case of
these Atari games, it is possible to choose a value of \\(\\lambda\\) such that
the extra term in the loss will have a noticeable effect on entropy value, while
final performance will not suffer.

BeamRider is different because for these particular training runs, none of the
agents achieved good performance. This is expected: in the original PPO paper it is reported that learning only starts after
roughly 10M frames (we terminate training after exactly 10M frames).

The following is a table with a summary of the above.

<div class="figure">
<table class="center">
<thead>
<tr class="header">
<th>\(\lambda\)</th>
<th>BeamRider</th> <th>Breakout</th> <th>Enduro</th> <th>Frostbite</th> <th>MsPacman</th> <th>Seaquest</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td>0.0</td> <td>540±30</td> <td>131±52</td> <td>930±155</td> <td>275±9</td> <td>2147±354</td> <td>851±15</td>
</tr>
<tr class="even">
<td>0.0005</td> <td>569±69</td> <td>120±52</td> <td>928±143</td> <td>273±15</td> <td>2052±336</td> <td>849±39</td>
</tr>
<tr class="odd">
<td>0.001</td> <td>547±46</td> <td>84±31</td> <td>900±115</td> <td>266±7</td> <td>1727±266</td> <td>931±151</td>
</tr>
<tr class="even">
<td>0.002</td> <td>537±43</td> <td>79±21</td> <td>654±87</td> <td>271±16</td> <td>1787±72</td> <td>787±36</td>
</tr>
<tr class="odd">
<td>0.003</td> <td>589±62</td> <td>84±15</td> <td>761±234</td> <td>270±9</td> <td>1546±134</td> <td>911±162</td>
</tr>
<tr class="even">
<td>0.005</td> <td>577±55</td> <td>52±6</td> <td>568±130</td> <td>278±14</td> <td>1554±308</td> <td>954±108</td>
</tr>
</tbody>
</table>
<p class="caption">Table 1: Summary of experiments.</p>
</div>

The following videos show what learned attention maps look like for different
\\(\\lambda\\).

<figure class="video_container">
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/BeamRider.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/Breakout.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/MsPacman.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/Frostbite.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/Enduro.mp4" type="video/webm">
</video>
<video controls="true" allowfullscreen="true", width="800">
    <source src="videos/Seaquest.mp4" type="video/webm">
</video>
<p class="caption">Figure 3: Gameplay video for agents trained with different values of \(\lambda\) (left to right: \(\lambda = 0, 0.0005, 0.001, 0.002, 0.003, 0.005\)). In each video, the top row shows the original observations with an attention overlay, and the bottom row shows the observations as received by the neural network after preprocessing. In the attention overlay, each rectangle corresponds to one neuron in the attention layer, and the color intensity is proportional to activation values.</p>
</figure>

### Conclusion

It is possible to apply entropy loss to the attention layer in RL agents in such way that it has a noticeable effect on visualizations, but performance does not deteriorate.

## References

<div class="footnotes">
<ol>
    <li id="fn1"><p>Yang et al. <a href="https://arxiv.org/pdf/1812.11276.pdf">Learn to Interpret Atari Agents</a><a href="#fnref1">↩</a></p></li>
    <li id="fn2"><p>Simonyan et al. <a href="https://arxiv.org/abs/1312.6034.pdf">Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps</a><a href="#fnref2">↩</a></p></li>
    <li id="fn3"><p>Hill et al. <a href="https://github.com/hill-a/stable-baselines">Stable Baselines</a><a href="#fnref3">↩</a></p></li>
    <li id="fn4"><p>Schulman et al. <a href="https://arxiv.org/abs/1707.06347">Proximal Policy Optimization Algorithms</a><a href="#fnref4">↩</a></p></li>
    <li id="fn5"><p>Smilkov et al. <a href="https://arxiv.org/abs/1706.03825">SmoothGrad: removing noise by adding noise</a><a href="#fnref5">↩</a></p></li>
    <li id="fn6"><p>Adebayo et al. <a href="https://arxiv.org/abs/1810.03307">Local Explanation Methods for Deep Neural Networks Lack Sensitivity to Parameter Values</a><a href="#fnref6">↩</a></p></li>
</ol>
</div>

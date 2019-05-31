# [PYTORCH] Asynchronous Advantage Actor-Critic (A3C) for playing Super Mario Bros

## Introduction

Here is my python source code for training an agent to play super mario bros. By using Asynchronous Advantage Actor-Critic (A3C) algorithm introduced in the paper **Asynchronous Methods for Deep Reinforcement Learning** [paper](https://arxiv.org/abs/1602.01783).
<p align="center">
  <img src="demo/video_1_1.gif">
  <img src="demo/video_1_2.gif">
  <img src="demo/video_1_4.gif"><br/>
  <img src="demo/video_2_3.gif">
  <img src="demo/video_3_1.gif">
  <img src="demo/video_3_4.gif"><br/>
  <img src="demo/video_4_1.gif">
  <img src="demo/video_6_1.gif">
  <img src="demo/video_7_1.gif"><br/>
  <i>Sample results</i>
</p>

## Motivation

Before I implemented this project, there are several repositories reproducing the paper's result quite well, in different common deep learning frameworks such as Tensorflow, Keras and Pytorch. In my opinion, most of them are great. However, they seem to be overly complicated in many parts including image's pre-processing, environtment setup and weight initialization, which distracts user's attention from more important matters. Therefore, I decide to write a cleaner code, which simplifies unimportant parts, while still follows the paper strictly. As you could see, with minimal setup and simple network's initialization, as long as you implement the algorithm correctly, agent(s) will teach itself how to interact with environment and gradually find out the way to reach the final goal.

## Explanation in layman's term
If you are already familiar to reinforcement learning in general and A3C in particular, you could skip this part. I write this part for explaining what is A3C algorithm, how and why it works, to people who are interested in or curious about A3C or my implementation, but do not understand the mechanism behind. Therefore, you do not need any prerequiste knowledge for reading this part :relaxed:

If you search on the internet, there are numerous article introducing or explaining A3C, some even provide sample code. However, I would like to take another approach: Break down the name **Asynchronous Actor-Critic Agents** into smaller parts and explain in an aggregated manner:

# Actor-Critic
Your agent has 2 parts called **actor** and **critic**, and its goal is to make both parts perfom better over time by exploring and exploiting the environment. Let imagine a small mischievous child (**actor**) is discovering the amazing world around him, while his dad (**critic**) oversees him, to make sure that he does not do anything dangerous. Whenever the kid does anything good, his dad will praise and encourage him to repeat that action in the future. And of course, when the kid do anything harmful, he will get warning from his dad. The more the kid interact to the world, and take different actions, the more feedback, both positive and negative, he gets from his dad. The goal of the kid is, to collect as many positive feedback as possible from his dad, while the goal of the dad is to evaluate his son's action better. In other word, we have a win-win relationshop between the kid and his dad, or equivalently between **actor** and **critic**.

## How to use my code

With my code, you can:
* **Train your model** by running **python train.py**
* **Test your trained model** by running **python test.py**

## Trained models

You could find my trained model at **trained_models/**
 
## Requirements

* **python 3.6**
* **gym**
* **cv2**
* **pytorch** 
* **numpy**

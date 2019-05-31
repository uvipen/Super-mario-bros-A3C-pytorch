# [PYTORCH] Asynchronous Actor-Critic Agents (A3C) for playing Super Mario Bros

## Introduction

Here is my python source code for training an agent to play super mario bros. By using Asynchronous Actor-Critic Agents (A3C) algorithm introduced in the paper **Asynchronous Methods for Deep Reinforcement Learning** [paper](https://arxiv.org/abs/1602.01783).
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

## How to use my code

With my code, you can:
* **Train your model from scratch** by running **python train.py**
* **Test your trained model** by running **python test.py**

## Trained models

You could find my trained model at **trained_models/**
 
## Requirements

* **python 3.6**
* **gym**
* **cv2**
* **pytorch** 
* **numpy**

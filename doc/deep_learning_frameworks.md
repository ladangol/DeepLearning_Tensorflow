# Deep Learning Frameworks

## Introduction
There are many libraries exists for building deep learning application. The most common ones are TensorFlow, Keras and PyTorch. These three libraries are all open-source.

## PyTorch
PyTorch is based on the Torch framework and originally developed by Facebook. PyTorch uses dynamic computational graphs. PyTorch is referred to as a 'define by run" framework.

## TensorFlow
TensorFlow is developed by Google. TensorFlow uses static computational graphs. Basically, TensorFlow computations are represented as a dataflow graph. TensorFlow is referred to as a 'define then run" framework. It is also good for distributed computing.

## Keras
Keras is a high-level Neural Networks library that is running on top of TensorFlow. It is modular and lets you build sequence-based and graph-based networks. Keras has many build-in algorithms such as optimizers, normalization, and activation layers.

## Dynamic vs. Static Computational Graph
Before comparing the three frameworks let's talk about the dynamic computational graph and static computational graph.

One advantage of the static computational graph to dynamic computational graph is efficiency and optimization. Framework optimized your graph before running it. This can improve execution time. However, this can make the process of defining a computational graph very complex and not straight forward.

static computational graphs are kind of like a variable in which it's memory is allocated on the stack. Dynamic computational graphs are like dynamic memory, that is the memory that is allocated on the heap.

Let's see the differences between having a variable in stack or heap. In order to be able to define a variable in the stack, at the time of definition you must also define how much memory it is required. On the other hand, when you are defining the dynamic variable, you do not need to specify the amount of memory allocation at the time of definition. You can postpone this process until the run time or execution time.

Dynamic memory allocation is valuable for situations where you cannot determine beforehand how much memory is required. Similarly, dynamic computational graphs are valuable for situations where you cannot determine the computation. One clear example of this is recursive computations that are based on variable data.

Debugging, image pre-processing before performing training, executing code on multi-GPUs, making new algorithms are much easier when we are using dynamic computation. For all the above reasons PyTorch is a preferable framework in academic environments.

## Comparison Factors
In terms of seed, Keras is the slowest of the three libraries. Tensorflow and PyTorch are having very similar and comparable performance. As a result of Keras is being the slowest of the three when we require to use high-performance models and large datasets that require fast execution we should consider switching to Tensorflow or PyTorch.
Among the three libraries, Tensorflow has the hardest architecture when comes to readability and ease of use. Keras can be considered the simplest one.
When comes to debugging Pythorch provides better-debugging capabilities compared to the other two. Tensorflow is the most challenging and difficult one to perform debugging.
There is not a clear formula or answer to choose one framework among all three however you can use the 3 following guide for your selection:
Technical background and skills
Project requirements
Ease of Use of the framework

I'm not really trying to recommend one framework over another but If you are a beginner in the land of deep learning I suggest that start you practise by using Keras library. In this case, you take advantage of the ease of use of Keras and the intensive support of TensorFlow. As a beginner, you are interested in building a deep learning model fast and you do not want to deal with the complexity of the framework itself.
If you absolutely have to use the TensorFlow framework, I suggest using Tensorflow version 2.0 or above since this version is based on they use eager execution. It will make the process of learning much easier and faster.

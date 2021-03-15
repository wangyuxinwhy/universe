# TorchScript 简介

> With TorchScript, PyTorch aims to create a unified framework from research to production. 

Pytorch 一直被人诟病的就是模型部署上的无能为力，由于其动态图的特性，使之没有办法满足生产环境中严苛的性能要求。为了解决这一问题，`TorchScript` 应运而生。

[TOC]

## 为什么要使用 TorchScript ?

1. 通用性，TorchScript 将模型推断的计算流程与 Python 代码解耦，使之不再依赖于 Python 解释器。这有利于将模型部署到更通用的计算设备上。
2. 高性能，TorchScipt 借助 Pytorch JIT 可以根据运行时信息对计算图进行自动优化。这与其他 JIT 如 XLA、numba 等工具类似。

## 如何将动态模型转换为 TorchScript ?

Pytorch 的 jit 模块主要提供了两个工具接口 `pytorch.jit.trace` 和 `pytorch.jit.script`。这两个 API 可以让用户方便地将动态模型转化为静态脚本，但两者均存在局限性，需要相互配合。

### torch.jit.trace

`torch.jit.trace` 接受一个函数或者 `nn.Module` 以及一个输入示例，通过追踪输入示例中 tensor 的变化来记录其计算流程。具体方式如下所示：

```python
import torch
import victor
from allennlp.models import load_archive
from allennlp.data import Batch
from mosch.dataset_reader import MoschTaggingDatasetReader

model_path = victor.repo.get_from_cache("mosch/rnn-base-model.tar.gz")
archive = load_archive(model_path)
instance = archive.dataset_reader.text_to_instance(["北京", "大学"])
instance.index_fields(archive.model.vocab)
tokens = Batch([instance]).as_tensor_dict()["tokens"]
# 传入 Module 以及 示例输入
mosch_script = torch.jit.trace(archive.model, (tokens, ), strict=False)
print(mosch_script.code)
```

OUTPUT:
```python
def forward(
  	self,
    argument_1: Dict[str, Dict[str, Tensor]]) -> Dict[str, Tensor]:
    _0 = self.tag_projection_layer
    _1 = self.encoder
    _2 = self.text_field_embedder
    tokens = (argument_1["tokens"])["tokens"]
    _3 = (_2).forward(tokens, )
    batch_size = ops.prim.NumToTensor(torch.size(_3, 0))
    _4 = int(batch_size)
    sequence_length = ops.prim.NumToTensor(torch.size(_3, 1))
    _5 = int(sequence_length)
    mask = torch.ne(tokens, 0)
    _6 = (_0).forward((_1).forward(mask, _3, ), )
    input = torch.view(_6, [-1, 5])
    _7 = torch.view(torch.softmax(input, -1, None), [_4, _5, 5])
    _8 = {"logits": _6, "class_probabilities": _7}
    return _8
```

`trace` 需要追踪输入示例 Tensor 的流动和变化，但有可能模型代码中存在分支代码在当前示例输入下没有运行的情况。这部分代码显然 `trace` 无力进行追踪，因此在此种情况下就无法使用 `trace` 来做转换。

上文的代码执行时，就会因为分支代码的问题产生一系列如下的 Warning:

```
pytorch_seq2seq_wrapper.py:117: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect.
```

### torch.jit.script

PyTorch 为了解决模型代码中存在分支结构的情况，由此提供了 `torch.jit.script` 来直接对 Python 源码进行分析，并将其编译转换为 TorchScipt。

具体使用方式如下：

```python
import torch

N = 10

class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 10:
            return x
        else:
            return -x

c = torch.jit.script(MyDecisionGate())

print(c.code)
```

OUTPUT

```python
def forward(self,
    x: Tensor) -> Tensor:
  _0 = torch.gt(torch.sum(x, dtype=None), 10)
  if bool(_0):
    _1 = x
  else:
    _1 = torch.neg(x)
  return _1
```

`scipt` 使用起来极为简单，它会自动扫描 python 的字面源码并进行转换，但不会对计算流程进行追踪。这因此带来了一个问题，因为 `scipt` 只扫描了一部分模型代码，当模型代码中存在全局变量或者调用了无法编译的第三方函数和包时，就会产生错误。

比如，我们将上方的代码改为通过全局变量来控制计算流程：

```python
import torch

N = 10
class MyDecisionGate(torch.nn.Module):
    def forward(self, x):
        if x.sum() > N:
            return x
        else:
            return -x

c = torch.jit.script(MyDecisionGate())
```

`torch.jit.script` 就会报错，如下：

```
RuntimeError: 
python value of type 'int' cannot be used as a value. Perhaps it is a closed over global variable? If so, please consider passing it in as an argument or use a local varible instead.
```

### trace 和 script 有哪些局限，如何互补 ？

`trace` 追踪计算流程，但无法处理分支结构。`script` 扫描 python 源码，但无法处理此代码块外的方法和变量。因此将两者结合在一起，才能完成正确的转换，如何将两者混合使用可以参考官方教程。**果然天下没有白吃的午餐。**

## TorchScript 如何存储和装载 ？

### save

```python
traced.save('wrapped_rnn.pt')
```

### load

```python
loaded = torch.jit.load('wrapped_rnn.pt')
```

## 如何部署 TorchScript ?

待实践

## 术语

- PyTorch JIT:  PyTorch JIT is an optimized compiler for PyTorch programs.
  - It is a lightweight threadsafe interpreter
  -  Supports easy to write custom transformations
  -  It’s not just for inference as it has auto diff support
- TorchScript: TorchScript is a static high-performance subset of Python language, specialized for ML applications. It supports 
	- Complex control flows
	- Common data structures User-defined classes
	- User-defined classes

## Reference

- [TorchScript 官方教程](https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html)
- [PyTorch JIT and TorchScript](https://towardsdatascience.com/pytorch-jit-and-torchscript-c2a77bac0fff)
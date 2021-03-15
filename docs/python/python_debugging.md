# Python Debugging

[TOC]

## 01 为什么要使用 python debugging tools ?

在代码运行中出现了 BUG 时，我们需要观察代码逻辑的运行和变量的数值变化，从而确定 BUG 产生的原因。python debugging tools 优雅的满足了我们的需求。

## 02 有哪些 python debugging tools 值得尝试？

参考 [best-of-python-dev](https://github.com/ml-tooling/best-of-python-dev#debugging-tools)，主要有以下三个工具：

- `pdb`  python 标准库自带 debug 工具
- `ipdb` 通过 ipython 进行交互式 debug
- `PySnooper` github 14.2k stars

下文主要介绍 `pdb` 和 `ipdb` 的使用方法，在非 IDE 开发环境中这两个工具都能不错的完成 debug 任务，不过如果能使用 PyCharm 的话，还是用 PyCharm 自带的 debug 工具吧。至于 `PySnooper` .... 我还没有找到合适的使用场景。

## 03 如何使用 `pdb` 或者 `ipdb` ?

### 0301 如何启用 `pdb` ？

在用 `PyTorch` 写 NLP 程序的时候，我们需要关注每一个 `nn.Module` 中参数的性质，比如 *shape* 等信息。我们可以显式的在代码中打断点，在 python3.7 之后可以直接键入 `breakpoint()` ，在 python3.7 之前可以使用 `import pdb;pdb.set_trace()`。具体代码如下所示：

```python
# debug_demo.py

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        breakpoint()
```

添加了断点之后，直接运行此程序即可。

```shell
python debug_demo.py
```

!!! note
		同样的我们可以不显式的打断点，可以直接通过 `python -m pdb debug_demo.py` 来进入 pdb 的交互界面，不过我觉得这种用法没啥必要。

### 0302 `pdb` 的常用命令 ？

- `s` 代表 `step into`，进入到下一次执行
- `n` 代表 `step over`，运行到下一行
- `c` 代表 `continue` ，继续运行代码直到下一个断点
- `p <exp>` 计算其后的表达式
- `b <filename>:<lineno>` 打断点
- `ll` 显示上下文代码
- `a` 显示此命名空间的变量值
- `q` 退出

命令即键入相应的代码，回车即可。

`RealPython` 整理了如下重要命令

| Command     | Description                                                  |
| ----------- | ------------------------------------------------------------ |
| `p`         | Print the value of an expression.                            |
| `pp`        | Pretty-print the value of an expression.                     |
| `n`         | Continue execution until the next line in the current function is reached or it returns. |
| `s`         | Execute the current line and stop at the first possible occasion (either in a function that is called or in the current function). |
| `c`         | Continue execution and only stop when a breakpoint is encountered. |
| `unt`       | Continue execution until the line with a number greater than the current one is reached. With a line number argument, continue execution until a line with a number greater or equal to that is reached. |
| `l`         | List source code for the current file. Without arguments, list 11 lines around the current line or continue the previous listing. |
| `ll`        | List the whole source code for the current function or frame. |
| `b`         | With no arguments, list all breaks. With a line number argument, set a breakpoint at this line in the current file. |
| `w`         | Print a stack trace, with the most recent frame at the bottom. An arrow indicates the current frame, which determines the context of most commands. |
| `u`         | Move the current frame count (default one) levels up in the stack trace (to an older frame). |
| `d`         | Move the current frame count (default one) levels down in the stack trace (to a newer frame). |
| `h`         | See a list of available commands.                            |
| `h <topic>` | Show help for a command or topic.                            |
| `h pdb`     | Show the full pdb documentation.                             |
| `q`         | Quit the debugger and exit.                                  |


### 0303 `pdb` QA

1. 如何让 debug 程序默认为 `ipdb` ?

设置环境变量 `PYTHONBREAKPOINT` 为 `ipdb.set_trace` 。

2. 如何在 `pdb` 中输入多行代码？

使用 `IPython` 的嵌入，运行 `from IPython import embed; embed()`

## 04 如何使用 PyCharm Debug ?

查看 PyCharm Debugger 的官方教程 [here](https://www.jetbrains.com/pycharm/features/debugger.html)

### 0401 Pycharm Debugger 特性

- 带有条件的 suspend
- `evaluate and log` 但是不挂起
- Run to cursor ，运行到鼠标指针处
- 断点管理
- 无侵入性代码
- 速度更快


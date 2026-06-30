# BUAA CST Digital Signal Processing

本仓库用于存放数字信号处理（Digital Signal Processing, DSP）课程实验材料，包含实验代码、LaTeX 报告源码、PDF 报告、实验图像和提交压缩包。

README 中的数学公式使用 GitHub 支持较稳定的 dollar math 写法，避免旧式 LaTeX 分隔符在仓库首页不渲染的问题。

## 仓库结构

```text
.
|-- README.md
|-- requirements.txt
|-- experiments/
|   |-- experiment1/
|   |   |-- experiment1.py
|   |   |-- 实验一报告.tex
|   |   |-- 实验一报告.pdf
|   |   |-- 2026 年春+傅粱训+24373088+实验一报告.pdf
|   |   |-- 数字信号处理实验一报告.md
|   |   |-- buaathesis.cls
|   |   |-- figure/
|   |   `-- figures/
|   `-- experiment2/
|       |-- experiment2.py
|       |-- 实验二报告.tex
|       |-- 实验二报告.pdf
|       |-- 2026 年春+傅粱训+24373088+实验二报告.pdf
|       |-- buaathesis.cls
|       |-- figure/
|       `-- figures/
`-- submission/
    `-- DSP1_傅粱训_24373088_实验一二提交.zip
```

两个实验目录都保留了各自编译 LaTeX 报告所需的 `buaathesis.cls`、校徽/校名资源和实验图像，因此可以单独进入对应实验目录运行代码或编译报告。

## 运行环境

代码使用 Python 完成，主要依赖见 `requirements.txt`：

- `numpy`
- `matplotlib`
- `scipy`

建议先安装依赖：

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

运行方式：

```powershell
cd experiments/experiment1
..\..\.venv\Scripts\python experiment1.py

cd ../experiment2
..\..\.venv\Scripts\python experiment2.py
```

如果系统中的 `python` 命令已正确配置，也可以直接使用 `python experiment1.py` 和 `python experiment2.py`。

实验一图像输出到 `experiments/experiment1/figures/`；实验二图像输出到 `experiments/experiment2/figures/experiment2/`。

## 实验一：时域采样与频域采样

### 实验要求

对模拟信号进行时域采样：

$$
x_a(t)=Ae^{-\alpha t}\sin(\Omega_0t)u(t)
$$

其中：

$$
A=444.128,\qquad
\alpha=50\sqrt{2}\pi,\qquad
\Omega_0=50\sqrt{2}\pi\ \mathrm{rad/s}
$$

取观测时间：

$$
T_p=50\ \mathrm{ms}
$$

分别使用三种采样频率：

$$
F_s=1000\ \mathrm{Hz},\quad 300\ \mathrm{Hz},\quad 200\ \mathrm{Hz}
$$

采样点数按下式计算：

$$
N=T_pF_s
$$

实验需要完成：

1. 生成三组采样序列。
2. 对三组采样序列统一作 64 点 FFT。
3. 绘制采样序列和幅频特性。
4. 分析采样频率不足时产生的频谱混叠现象。
5. 构造长度为 $M=27$ 的有限长三角序列：

$$
x(n)=
\begin{cases}
n+1, & 0\le n\le 13,\\
27-n, & 14\le n\le 26,\\
0, & \text{其他}.
\end{cases}
$$

6. 对 $X(e^{j\omega})$ 在 $[0,2\pi]$ 上分别进行 32 点和 16 点等间隔频域采样。
7. 分别作 IFFT，比较 $x(n)$、$x_{32}(n)$、$x_{16}(n)$，验证频域采样导致时域周期延拓的规律。
8. 思考当频域采样点数 $N<M$ 时，如何用一次最少点数的 DFT 得到所需频谱采样。

### 核心结论

时域采样会导致频域周期延拓，采样频率越低，频谱混叠越明显；频域采样会导致时域周期延拓，当采样点数小于原序列长度时会发生时域混叠。

## 实验二：IIR 数字滤波器设计及软件实现

### 实验要求

采样频率和采样点数为：

$$
F_s=10000\ \mathrm{Hz},\qquad N=1600
$$

产生三路抑制载波单频调幅（SCAM）信号，并将其相加得到复合信号 $s(t)$。三路信号参数为：

| 信号 | 载波频率 $f_c$ / Hz | 调制频率 $f_m$ / Hz | 差频 / Hz | 和频 / Hz |
| --- | ---: | ---: | ---: | ---: |
| $x_1(t)$ | 1000 | 100 | 900 | 1100 |
| $x_2(t)$ | 500 | 50 | 450 | 550 |
| $x_3(t)$ | 250 | 25 | 225 | 275 |

实验需要完成：

1. 绘制复合信号 $s(t)$ 的时域波形和 FFT 幅频特性。
2. 观察六根主要谱线：

$$
225,\ 275,\ 450,\ 550,\ 900,\ 1100\ \mathrm{Hz}
$$

3. 设计低通、带通、高通三个椭圆 IIR 数字滤波器，用于分离三路信号。
4. 滤波器指标为：

$$
R_p=0.1\ \mathrm{dB},\qquad R_s=60\ \mathrm{dB}
$$

5. 三个滤波器的通阻带指标如下：

| 滤波器 | 分离信号 | 通带频率 / Hz | 阻带频率 / Hz |
| --- | --- | --- | --- |
| 低通 | $y_3(n)$ | $0\sim 300$ | $400\sim 5000$ |
| 带通 | $y_2(n)$ | $400\sim 600$ | $0\sim 350,\ 700\sim 5000$ |
| 高通 | $y_1(n)$ | $800\sim 5000$ | $0\sim 700$ |

6. 计算并绘制三个滤波器的损耗函数，检查通带最大衰减和阻带最小衰减是否满足题目要求。
7. 用三个滤波器分别对复合信号滤波，绘制分离后信号的时域波形和幅频特性，分析频域分离效果。
8. 思考改变采样点数时是否仍能得到 6 根理想谱线，并比较加入载波成分后的 AM 信号与 SCAM 信号在时域和频域上的差别。

### 核心结论

三路抑制载波调幅信号在时域混合后难以直接分辨，但其频谱互不重叠，可以通过低通、带通、高通 IIR 滤波器在频域中分离。椭圆滤波器能在较低阶数下满足 $0.1\ \mathrm{dB}$ 通带衰减和 $60\ \mathrm{dB}$ 阻带衰减要求。

## 报告编译说明

LaTeX 报告使用 `buaathesis.cls` 模板，建议使用 XeLaTeX 编译。进入对应实验目录后编译：

```bash
cd experiments/experiment1
xelatex 实验一报告.tex

cd ../experiment2
xelatex 实验二报告.tex
```

编译时需保留同目录下的 `buaathesis.cls`、`figure/` 和 `figures/`，否则模板或图片资源可能无法找到。

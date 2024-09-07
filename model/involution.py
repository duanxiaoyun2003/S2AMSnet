from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn as nn
from mmcv.cnn import ConvModule

from collections import namedtuple
import cupy
from string import Template

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS

#基于输入特征图 bottom_data 和动态生成的 weight_data，
#在每个输出位置执行自定义卷积操作。每个 CUDA 线程计算输出张量中的一个位置值，
#并将计算结果写回到 top_data 中。
_involution_kernel = kernel_loop + '''   #这是一个 CUDA 内核函数，用于在 GPU 上并行执行 involution 操作
extern "C"
__global__ void involution_forward_kernel(
const ${Dtype}* bottom_data, const ${Dtype}* weight_data, ${Dtype}* top_data) {  
#bottom_data:输入特征图数据（图像）。
#weight_data:权重数据(kernel weight)。
#top_data:输出数据（结果张量）。
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
  #CUDA_KERNEL_LOOP 用于在 CUDA 线程中执行循环,index 是当前线程的全局索引,nthreads 表示需要处理的总线程数。
    const int n = index / ${channels} / ${top_height} / ${top_width};
    const int c = (index / ${top_height} / ${top_width}) % ${channels};
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    #通过 index 计算当前元素所在的批次(n)、通道(c)、高度(h)和宽度(w)。这些值用于定位当前线程需要处理的输出张量位置。
    const int g = c / (${channels} / ${groups});
    #g 表示当前通道所属的组。通过 channels / groups 的比率来确定通道在哪个组内。

    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {  #双重循环 kh 和 kw 用于遍历内核的高度和宽度。
        const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
        const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
        #h_in 和 w_in 计算输入图像中卷积核覆盖的像素位置，
        #考虑了填充 (pad_h, pad_w)、步幅 (stride_h, stride_w) 和膨胀 (dilation_h, dilation_w)。
        if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
          #检查 h_in 和 w_in 是否在输入特征图的有效范围内，确保不会访问越界内存。
          const int offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
            * ${bottom_width} + w_in;
            
          const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h)
            * ${top_width} + w;
            ##通过批次、通道、高度、宽度等信息计算 bottom_data 和 weight_data 的位置偏移量 (offset 和 offset_weight)。

          value += weight_data[offset_weight] * bottom_data[offset];
          #乘以对应的 weight_data 和 bottom_data 的值，并累加到 value 中，这个值最终会成为 top_data 中对应位置的卷积输出结果。
        }
      }
    }
    top_data[index] = value;
    #将计算得到的 value 写入输出张量 top_data 对应的索引位置 index。
  }
}
'''

_involution_kernel_backward_grad_input = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_input_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const weight_data, ${Dtype}* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int n = index / ${channels} / ${bottom_height} / ${bottom_width};
    const int c = (index / ${bottom_height} / ${bottom_width}) % ${channels};
    const int h = (index / ${bottom_width}) % ${bottom_height};
    const int w = index % ${bottom_width};
    const int g = c / (${channels} / ${groups});
    ${Dtype} value = 0;
    #pragma unroll
    for (int kh = 0; kh < ${kernel_h}; ++kh) {
      #pragma unroll
      for (int kw = 0; kw < ${kernel_w}; ++kw) {
        const int h_out_s = h + ${pad_h} - kh * ${dilation_h};
        const int w_out_s = w + ${pad_w} - kw * ${dilation_w};
        if (((h_out_s % ${stride_h}) == 0) && ((w_out_s % ${stride_w}) == 0)) {
          const int h_out = h_out_s / ${stride_h};
          const int w_out = w_out_s / ${stride_w};
          if ((h_out >= 0) && (h_out < ${top_height})
                && (w_out >= 0) && (w_out < ${top_width})) {
            const int offset = ((n * ${channels} + c) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            const int offset_weight = ((((n * ${groups} + g) * ${kernel_h} + kh) * ${kernel_w} + kw) * ${top_height} + h_out)
                  * ${top_width} + w_out;
            value += weight_data[offset_weight] * top_diff[offset];
          }
        }
      }
    }
    bottom_diff[index] = value;
  }
}
'''

_involution_kernel_backward_grad_weight = kernel_loop + '''
extern "C"
__global__ void involution_backward_grad_weight_kernel(
    const ${Dtype}* const top_diff, const ${Dtype}* const bottom_data, ${Dtype}* const buffer_data) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int h = (index / ${top_width}) % ${top_height};
    const int w = index % ${top_width};
    const int kh = (index / ${kernel_w} / ${top_height} / ${top_width})
          % ${kernel_h};
    const int kw = (index / ${top_height} / ${top_width}) % ${kernel_w};
    const int h_in = -${pad_h} + h * ${stride_h} + kh * ${dilation_h};
    const int w_in = -${pad_w} + w * ${stride_w} + kw * ${dilation_w};
    if ((h_in >= 0) && (h_in < ${bottom_height})
          && (w_in >= 0) && (w_in < ${bottom_width})) {
      const int g = (index / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${groups};
      const int n = (index / ${groups} / ${kernel_h} / ${kernel_w} / ${top_height} / ${top_width}) % ${num};
      ${Dtype} value = 0;
      #pragma unroll
      for (int c = g * (${channels} / ${groups}); c < (g + 1) * (${channels} / ${groups}); ++c) {
        const int top_offset = ((n * ${channels} + c) * ${top_height} + h)
              * ${top_width} + w;
        const int bottom_offset = ((n * ${channels} + c) * ${bottom_height} + h_in)
              * ${bottom_width} + w_in;
        value += top_diff[top_offset] * bottom_data[bottom_offset];
      }
      buffer_data[index] = value;
    } else {
      buffer_data[index] = 0;
    }
  }
}
'''


class _involution(Function):
    @staticmethod    #定义静态方法
    def forward(ctx, input, weight, stride, padding, dilation):
        assert input.dim() == 4 and input.is_cuda
        assert weight.dim() == 6 and weight.is_cuda
        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]   #获取 weight 的卷积核高度和宽度。
        output_h = int((height + 2 * padding[0] - (dilation[0] * (kernel_h - 1) + 1)) / stride[0] + 1)
        output_w = int((width + 2 * padding[1] - (dilation[1] * (kernel_w - 1) + 1)) / stride[1] + 1)
        #根据输入形状、填充、步长和膨胀系数，计算输出张量的高度和宽度。

        output = input.new(batch_size, channels, output_h, output_w)
        #初始化输出张量 output，与输入数据类型相同。
        n = output.numel()  #用于计算张量 output 中的总元素数量。
        # n=batch_size*channels*output_h*output_w

        with torch.cuda.device_of(input):   #确保在 input 张量所在的 CUDA 设备上执行后续的操作。
            f = load_kernel('involution_forward_kernel', _involution_kernel, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, groups=weight.size()[1],
                            bottom_height=height, bottom_width=width,
                            top_height=output_h, top_width=output_w,
                            kernel_h=kernel_h, kernel_w=kernel_w,
                            stride_h=stride[0], stride_w=stride[1],
                            dilation_h=dilation[0], dilation_w=dilation[1],
                            pad_h=padding[0], pad_w=padding[1])
            #load_kernel：加载名为 involution_forward_kernel 的自定义 CUDA 内核，用于执行前向传播的核心计算。
            #Dtype=Dtype(input)：指定数据类型与 input 一致。
            #nthreads=n：指定线程数，设置为 output.numel()，即输出张量中元素的数量。
            #其他参数如 num=batch_size、channels、kernel_h、stride_h 等用于定义卷积的形状、步幅、填充、膨胀等操作的具体计算。

            #启动 CUDA 内核执行
            f(block=(CUDA_NUM_THREADS, 1, 1),  #定义每个 CUDA block 中的线程数 CUDA_NUM_THREADS = 1024
              grid=(GET_BLOCKS(n), 1, 1),  #定义 CUDA grid 的大小，其中 GET_BLOCKS(n) 会根据元素数量 n 来计算需要的 block 数量，确保每个元素都能被分配到一个线程。
              args=[input.data_ptr(), weight.data_ptr(), output.data_ptr()],#将 input、weight 和 output 的数据指针传递给 CUDA 内核，用于执行卷积操作。
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))#指定 CUDA 流，确保内核在当前流中执行。流是一种并行化机制，用于在不同操作之间协调。

        ctx.save_for_backward(input, weight)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation
        return output

        #在前向传播中，根据动态生成的卷积核和输入张量计算卷积结果；

    @staticmethod
    def backward(ctx, grad_output):
        # assert grad_output.is_cuda and grad_output.is_contiguous()
        input, weight = ctx.saved_tensors
        stride, padding, dilation = ctx.stride, ctx.padding, ctx.dilation

        batch_size, channels, height, width = input.size()
        kernel_h, kernel_w = weight.size()[2:4]
        output_h, output_w = grad_output.size()[2:]

        grad_input, grad_weight = None, None

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels, groups=weight.size()[1],
                   bottom_height=height, bottom_width=width,
                   top_height=output_h, top_width=output_w,
                   kernel_h=kernel_h, kernel_w=kernel_w,
                   stride_h=stride[0], stride_w=stride[1],
                   dilation_h=dilation[0], dilation_w=dilation[1],
                   pad_h=padding[0], pad_w=padding[1])

        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_input = input.new(input.size())

                n = grad_input.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_input_kernel',
                                _involution_kernel_backward_grad_input, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), weight.data_ptr(), grad_input.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

            if ctx.needs_input_grad[1]:
                grad_weight = weight.new(weight.size())

                n = grad_weight.numel()
                opt['nthreads'] = n

                f = load_kernel('involution_backward_grad_weight_kernel',
                                _involution_kernel_backward_grad_weight, **opt)
                f(block=(CUDA_NUM_THREADS, 1, 1),
                  grid=(GET_BLOCKS(n), 1, 1),
                  args=[grad_output.data_ptr(), input.data_ptr(), grad_weight.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        return grad_input, grad_weight, None, None, None


def _involution_cuda(input, weight, bias=None, stride=1, padding=0, dilation=1):
    """ involution kernel
    """
    assert input.size(0) == weight.size(0)
    assert input.size(-2) // stride == weight.size(-2)
    assert input.size(-1) // stride == weight.size(-1)
    if input.is_cuda:
        out = _involution.apply(input, weight, _pair(stride), _pair(padding), _pair(dilation))
        #调用_involution类的前向传播方法
        if bias is not None:
            out += bias.view(1, -1, 1, 1)
    else:
        raise NotImplementedError
    return out


class involution(nn.Module):

    def __init__(self,
                 channels,
                 kernel_size,
                 stride):
        super(involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.channels = channels
        reduction_ratio = 5
        self.group_channels = 3
        self.groups = self.channels // self.group_channels
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=channels // reduction_ratio,
            kernel_size=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU'))
        if self.groups == 0:
            self.groups = 1
        self.conv2 = ConvModule(
            in_channels=channels // reduction_ratio,
            out_channels=kernel_size ** 2 * self.groups,
            kernel_size=1,
            stride=1)
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)

    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size, self.kernel_size, h, w)
        out = _involution_cuda(x, weight, stride=self.stride, padding=(self.kernel_size - 1) // 2)
        return out
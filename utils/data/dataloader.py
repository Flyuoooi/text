# refer to: https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/data/data_utils.py

import torch
import threading
import queue
from torch.utils.data import DataLoader
from torch import distributed as dist


"""
#based on http://stackoverflow.com/questions/7323664/python-generator-pre-fetch
This is a single-function package that transforms arbitrary generator into a background-thead generator that 
prefetches several batches of data in a parallel background thead.

This is useful if you have a computationally heavy process (CPU or GPU) that 
iteratively processes minibatches from the generator while the generator 
consumes some other resource (disk IO / loading from database / more CPU if you have unused cores). 

By default these two processes will constantly wait for one another to finish. If you make generator work in 
prefetch mode (see examples below), they will work in parallel, potentially saving you your GPU time.
We personally use the prefetch generator when iterating minibatches of data for deep learning with PyTorch etc.

Quick usage example (ipython notebook) - https://github.com/justheuristic/prefetch_generator/blob/master/example.ipynb
This package contains this object
 - BackgroundGenerator(any_other_generator[,max_prefetch = something])
"""


class BackgroundGenerator(threading.Thread):
    """
    the usage is below
    >> for batch in BackgroundGenerator(my_minibatch_iterator):
    >>    doit()
    More details are written in the BackgroundGenerator doc
    >> help(BackgroundGenerator)
    """

    def __init__(self, generator, local_rank, max_prefetch=10):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.

        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may raise GIL and zero-out the
        benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it
        outputs is passed through queue.

        There's no restriction on doing weird stuff, reading/writing files, retrieving
        URLs [or whatever] wlilst iterating.

        :param max_prefetch: defines, how many iterations (at most) can background generator keep
        stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until
        one of these batches is dequeued.

        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!

        Setting max_prefetch to -1 lets it store as many batches as it can, which will work
        slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size
        unless dequeued quickly enough.
        """
        super().__init__()
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.exit_event = threading.Event()
        self.start()

    def run(self):
        """后台线程的主函数"""
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            if self.exit_event.is_set():
                break
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        """获取下一个数据批次"""
        next_item = self.queue.get()  # 从队列获取数据（阻塞操作）
        if next_item is None:  # 遇到结束标志
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    # """增强的DataLoader，支持数据预取和异步GPU传输"""
    # def __init__(self, **kwargs):
    #     super().__init__(**kwargs)
    #     try:
    #         if dist.is_available() and dist.is_initialized():
    #             local_rank = dist.get_rank()
    #         else:
    #             local_rank = 0  # 单卡默认 rank=0
    #     except:
    #         local_rank = 0
    #     # 在每个进程中创建一个新的CUDA流，用于异步数据传输
    #     self.stream = torch.cuda.Stream(local_rank)  # create a new cuda stream in each process
    #     self.local_rank = local_rank
    #
    # # def __iter__(self):
    # #     """创建迭代器，并用BackgroundGenerator包装"""
    # #     self.iter = super().__iter__()  # 获取父类的迭代器
    # #     # 用后台生成器包装原始迭代器
    # #     self.iter = BackgroundGenerator(self.iter, self.local_rank)
    # #     self.preload()  # 预加载第一个批次
    # #     return self
    #
    # def __iter__(self):
    #     for batch in super().__iter__():
    #         yield batch
    #
    # def _shutdown_background_thread(self):
    #     """优雅关闭后台线程"""
    #     if not self.iter.is_alive():
    #         # avoid re-entrance or ill-conditioned thread state# 避免重入或线程状态不良
    #         return
    #
    #     # Set exit event to True for background threading stopping# 设置退出事件为True，通知后台线程停止
    #     self.iter.exit_event.set()
    #
    #     # Exhaust all remaining elements, so that the queue becomes empty,
    #     # and the thread should quit  # 耗尽所有剩余元素，使队列变空，线程应该退出
    #     for _ in self.iter:
    #         pass
    #
    #     # Waiting for background thread to quit # 等待后台线程退出
    #     self.iter.join()
    #
    # def preload(self):
    #     """预加载下一个数据批次到GPU"""
    #     self.batch = next(self.iter, None)   # 获取下一个批次
    #     if self.batch is None:
    #         return None
    #
    #     # 在指定的CUDA流中进行异步数据传输
    #     with torch.cuda.stream(self.stream):
    #         # if isinstance(self.batch[0], torch.Tensor):
    #         #     self.batch[0] = self.batch[0].to(device=self.local_rank, non_blocking=True)
    #         for k, v in enumerate(self.batch):  # 遍历批次中的所有元素
    #             if isinstance(self.batch[k], torch.Tensor):
    #                 # 将张量异步传输到指定GPU设备# 非阻塞传输，不等待完成
    #                 self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)
    #
    # def __next__(self):
    #     """获取下一个数据批次"""
    #     torch.cuda.current_stream().wait_stream(
    #         self.stream
    #     )  # wait tensor to put on GPU
    #     batch = self.batch
    #     if batch is None:
    #         raise StopIteration
    #     self.preload()  # 预加载下一个批次
    #     return batch
    #
    # # Signal for shutting down background thread# 关闭后台线程的信号
    # def shutdown(self):
    #     """关闭数据加载器，释放后台线程资源"""
    #     # If the dataloader is to be freed, shutdown its BackgroundGenerator # 如果要释放dataloader，关闭其BackgroundGenerator
    #     self._shutdown_background_thread()

    pass

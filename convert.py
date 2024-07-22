import os

# 设置环境变量，避免在Surya中使用多进程，避免在pdftext中使用多进程
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # 由于某种原因，transformers决定使用.isin进行简单的操作，这在MPS上不受支持
os.environ["IN_STREAMLIT"] = "true" # 避免在Surya中使用多进程
os.environ["PDFTEXT_CPU_WORKERS"] = "1" # 避免在pdftext中使用多进程

import pypdfium2 # 需要放在顶部以避免警告
import argparse
import torch.multiprocessing as mp
from tqdm import tqdm
import math

# 导入需要的模块
from marker.convert import convert_single_pdf
from marker.output import markdown_exists, save_markdown
from marker.pdf.utils import find_filetype
from marker.pdf.extract_text import get_length_of_text
from marker.models import load_all_models
from marker.settings import settings
from marker.logger import configure_logging
import traceback
import json

# 配置日志
configure_logging()


def worker_init(shared_model):
    # 如果共享模型为空，则加载所有模型
    if shared_model is None:
        shared_model = load_all_models()

    # 将共享模型赋值给全局变量
    global model_refs
    model_refs = shared_model


def worker_exit():
    # 删除全局变量
    global model_refs
    del model_refs


def process_single_pdf(args):
    # 处理单个pdf文件
    filepath, out_folder, metadata, min_length = args

    # 获取文件名
    fname = os.path.basename(filepath)
    # 如果markdown文件已经存在，则返回
    if markdown_exists(out_folder, fname):
        return

    try:
        # 跳过尝试转换嵌入文本很少的文件
        # 这可以表明它们被扫描，而不是正确地进行OCR
        # 通常这些文件不是最近的高质量文件
        if min_length:
            filetype = find_filetype(filepath)
            if filetype == "other":
                return 0

            length = get_length_of_text(filepath)
            if length < min_length:
                return

        # 转换pdf文件
        full_text, images, out_metadata = convert_single_pdf(filepath, model_refs, metadata=metadata)
        # 如果转换后的文本不为空，则保存markdown文件
        if len(full_text.strip()) > 0:
            save_markdown(out_folder, fname, full_text, images, out_metadata)
        else:
            print(f"空文件：{filepath}。无法转换。")
    except Exception as e:
        print(f"转换{filepath}时出错：{e}")
        print(traceback.format_exc())


def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="将多个pdf转换为markdown。")
    parser.add_argument("in_folder", help="包含pdf文件的输入文件夹。")
    parser.add_argument("out_folder", help="输出文件夹")
    parser.add_argument("--chunk_idx", type=int, default=0, help="要转换的块索引")
    parser.add_argument("--num_chunks", type=int, default=1, help="并行处理的块数量")
    parser.add_argument("--max", type=int, default=None, help="要转换的最大pdf数量")
    parser.add_argument("--workers", type=int, default=5, help="使用的worker进程数量")
    parser.add_argument("--metadata_file", type=str, default=None, help="用于过滤的元数据json文件")
    parser.add_argument("--min_length", type=int, default=None, help="要转换的pdf的最小长度")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取输入文件夹和输出文件夹的绝对路径
    in_folder = os.path.abspath(args.in_folder)
    out_folder = os.path.abspath(args.out_folder)
    # 获取输入文件夹中的所有文件
    files = [os.path.join(in_folder, f) for f in os.listdir(in_folder)]
    files = [f for f in files if os.path.isfile(f)]
    # 创建输出文件夹
    os.makedirs(out_folder, exist_ok=True)

    # 如果并行处理，确保所有文件都进入一个块
    # 计算块大小
    chunk_size = math.ceil(len(files) / args.num_chunks)
    # 计算起始索引和结束索引
    start_idx = args.chunk_idx * chunk_size
    end_idx = start_idx + chunk_size
    # 获取要转换的文件列表
    files_to_convert = files[start_idx:end_idx]

    # 如果需要，限制转换的文件数量
    if args.max:
        files_to_convert = files_to_convert[:args.max]

    # 如果提供了元数据文件，则加载元数据
    metadata = {}
    if args.metadata_file:
        metadata_file = os.path.abspath(args.metadata_file)
        with open(metadata_file, "r") as f:
            metadata = json.load(f)

    # 计算总进程数
    total_processes = min(len(files_to_convert), args.workers)

    # 根据GPU内存动态设置每个任务的GPU分配
    if settings.CUDA:
        tasks_per_gpu = settings.INFERENCE_RAM // settings.VRAM_PER_TASK if settings.CUDA else 0
        total_processes = int(min(tasks_per_gpu, total_processes))
    else:
        total_processes = int(total_processes)

    # 设置进程启动方法
    try:
        mp.set_start_method('spawn') # 对于CUDA，forkserver不起作用
    except RuntimeError:
        raise RuntimeError("将启动方法设置为spawn两次。这可能是一个临时问题。请再次运行脚本。")

    # 如果使用MPS，则无法使用torch多进程共享内存。这将使事情更少内存高效。如果您想共享内存，您必须使用CUDA或CPU。  设置TORCH_DEVICE环境变量以更改设备。
    if settings.TORCH_DEVICE == "mps" or settings.TORCH_DEVICE_MODEL == "mps":
        print("无法使用MPS与torch多进程共享内存。这将使事情更少内存高效。如果您想共享内存,您必须使用CUDA或CPU。  设置TORCH_DEVICE环境变量以更改设备。")

        model_lst = None
    else:
        # 加载所有模型
        model_lst = load_all_models()

        # 将模型共享内存
        for model in model_lst:
            if model is None:
                continue
            model.share_memory()

    # 打印转换信息
    print(f"在块{args.chunk_idx + 1}/{args.num_chunks}中转换{len(files_to_convert)}个pdf,使用{total_processes}个进程，并存储在{out_folder}")
    # 创建任务参数
    task_args = [(f, out_folder, metadata.get(os.path.basename(f)), args.min_length) for f in files_to_convert]

    # 使用多进程池处理任务
    with mp.Pool(processes=total_processes, initializer=worker_init, initargs=(model_lst,)) as pool:
        list(tqdm(pool.imap(process_single_pdf, task_args), total=len(task_args), desc="处理PDFs", unit="pdf"))

        # 终止worker处理程序
        pool._worker_handler.terminate = worker_exit

    # 删除所有CUDA张量
    del model_lst


if __name__ == "__main__":
    main()
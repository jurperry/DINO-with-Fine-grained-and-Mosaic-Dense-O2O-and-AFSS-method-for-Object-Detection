# 获取的文件可能有空txt数据
# 本文件用于剔除空txt文件, 然后根据txt文件名获取对应图片
# 并把图片与txt文件分别存储到指定文件夹内
import os
import shutil
from tqdm import tqdm

def find_empty_txt_files(folder_path, output_file="empty_files.txt"):
    """
    查找指定文件夹中内容为空的 txt 文件
    
    参数:
        folder_path: 要搜索的文件夹路径
        output_file: 输出结果的文件名
    """
    empty_files = []
    
    pbar = tqdm(total=len(os.listdir(folder_path)), desc="Process annotation files")
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            
            # 检查文件大小是否为0
            if os.path.getsize(file_path) == 0:
                empty_files.append(filename)
                continue
            
            # 检查文件内容是否只有空白字符
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()  # 去除首尾空白字符
                if not content:  # 如果内容为空
                    empty_files.append(filename)
        pbar.update(1)
    pbar.close()
    
    # 输出结果
    if empty_files:
        print(f"Found {len(empty_files)} empty TXT files.")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(empty_files))
        print(f"Empty file list saved to: {output_file}")
    else:
        with open(output_file, 'w') as f:
            pass  # 不写入任何内容，直接关闭文件
        print(f"Not found empty TXT files. Empty file list saved to: {output_file}")
        
# 本文件可以实现yolo格式下的空文件清洗, 把img, txt空文件清除, 并写入新的文件夹内
def filter_dataset(empty_txt_list, src_txt_dir, src_img_dir, dst_txt_dir, dst_img_dir):
    """
    根据空TXT文件列表过滤数据集
    
    参数:
        empty_txt_list: 空TXT文件名列表
        src_txt_dir: 源TXT文件夹路径
        src_img_dir: 源图片文件夹路径
        dst_txt_dir: 目标TXT文件夹路径
        dst_img_dir: 目标图片文件夹路径
    """
    # 创建目标目录
    os.makedirs(dst_txt_dir, exist_ok=True)
    os.makedirs(dst_img_dir, exist_ok=True)
    
    # 将空文件名列表转换为集合（不含扩展名）
    empty_files = {os.path.splitext(f)[0] for f in empty_txt_list}
    print(f"Found {len(empty_files)} empty annotation files.")
    
    # 处理TXT文件
    txt_files = [f for f in os.listdir(src_txt_dir) if f.endswith('.txt')]
    txt_count = 0
    
    for txt_file in tqdm(txt_files, desc="Process annotation files"):
        base_name = os.path.splitext(txt_file)[0]
        
        # 如果不是空文件则复制
        if base_name not in empty_files:
            src_path = os.path.join(src_txt_dir, txt_file)
            dst_path = os.path.join(dst_txt_dir, txt_file)
            shutil.copy2(src_path, dst_path)
            txt_count += 1
    
    # 处理图片文件
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    img_files = [f for f in os.listdir(src_img_dir) 
                if os.path.splitext(f)[1].lower() in img_extensions]
    img_count = 0
    
    for img_file in tqdm(img_files, desc="Process images"):
        base_name = os.path.splitext(img_file)[0]
        
        # 如果有对应的非空TXT文件则复制
        if base_name not in empty_files:
            src_path = os.path.join(src_img_dir, img_file)
            dst_path = os.path.join(dst_img_dir, img_file)
            shutil.copy2(src_path, dst_path)
            img_count += 1
    
    print(f"\nProcess completed:")
    print(f" - Copy {txt_count} non-empty annotation files to {dst_txt_dir}")
    print(f" - Copy {img_count} images to {dst_img_dir}")
    print(f" - Remove {len(empty_files)} empty annotation files and their corresponding images")

def empty_filter(src_img_dir, src_txt_dir, dst_img_dir, dst_txt_dir, empty_list_file):
    """
    过滤空标注文件, 并复制非空文件到目标文件夹
    参数:
        src_img_dir: 源图片文件夹路径
        src_txt_dir: 源TXT文件夹路径
        dst_img_dir: 目标图片文件夹路径
        dst_txt_dir: 目标TXT文件夹路径
        empty_list_file: 空文件列表txt路径
    """
    find_empty_txt_files(src_txt_dir, output_file=empty_list_file)
    with open(empty_list_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    empty_txt_list = [line.strip() for line in lines if line.strip()]
    if not lines:
        print("No empty txt files found.")
        return 0
    else:
        filter_dataset(empty_txt_list, src_txt_dir, src_img_dir, dst_txt_dir, dst_img_dir)
        return 1

if __name__ == "__main__":
    src_img_dir = "your_img_path" # 源图片文件夹
    src_txt_dir = "your_txt_path"   # 源TXT文件夹
    
    dst_img_dir = "your_output_img_path" # 目标图片文件夹
    dst_txt_dir = "your_output_txt_path"   # 目标TXT文件夹

    empty_list_file = "your_path/empty_files.txt" # 需要生成的空文件txt路径

    empty_filter(src_img_dir, src_txt_dir, dst_img_dir, dst_txt_dir, empty_list_file)
李沐 - 动手学深度学习，学习笔记
# ! py版本3.11 非特殊不要改动requirement.txt 否则会有包之间的版本约束冲突问题

# 开启虚拟环境 安装依赖包
python3.11 -m venv .venv
source .venv/bin/activate
python3.11 -m pip install -r requirement.txt

# 验证
python3.11 ch03-多层感知机/03.py
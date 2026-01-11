"""
ReAct Agent 实现
基于 ReAct (Reasoning + Acting) 模式的智能代理，能够通过思考-行动-观察的循环来完成任务。
"""

import ast  # 用于安全地解析 Python 字面量表达式
import inspect  # 用于获取函数签名和文档字符串
import os  # 用于文件系统操作
import re  # 用于正则表达式匹配
from string import Template  # 用于字符串模板替换
from typing import List, Callable, Tuple  # 类型提示

import click  # 用于命令行接口
from dotenv import load_dotenv  # 用于加载环境变量
from openai import OpenAI  # OpenAI API 客户端
import platform  # 用于获取操作系统信息

from prompt_template import react_system_prompt_template  # ReAct 系统提示词模板


class ReActAgent:
    """
    ReAct Agent 类
    
    实现基于 ReAct (Reasoning + Acting) 模式的智能代理。
    通过思考(Thought)、行动(Action)、观察(Observation)的循环来完成任务。
    """
    
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        """
        初始化 ReAct Agent
        
        Args:
            tools: 可用工具函数列表，Agent 可以调用这些工具来执行操作
            model: 使用的 LLM 模型名称
            project_directory: 项目目录路径，用于文件操作
        """
        # 将工具函数列表转换为字典，以函数名为键，方便查找
        self.tools = { func.__name__: func for func in tools }
        self.model = model  # LLM 模型名称
        self.project_directory = project_directory  # 项目目录路径
        # 初始化 OpenAI 客户端，连接到 SiliconFlow API
        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            #base_url="https://openrouter.ai/api/v1",
            api_key=ReActAgent.get_api_key(),
        )

    def run(self, user_input: str):
        """
        运行 Agent，执行用户任务
        
        这是 ReAct 循环的核心方法：
        1. Thought: 模型思考如何解决问题
        2. Action: 模型决定执行哪个工具
        3. Observation: 执行工具并观察结果
        4. 重复上述过程，直到得到最终答案
        
        Args:
            user_input: 用户输入的任务描述
            
        Returns:
            最终答案字符串
        """
        # 初始化消息列表，包含系统提示和用户问题
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        # ReAct 循环：持续执行思考-行动-观察，直到得到最终答案
        while True:

            # 步骤 1: 请求模型生成响应（包含 Thought 和 Action）
            content = self.call_model(messages)
            #print(f"******* The LLM Response Content: {content} \n*********")

            # 步骤 2: 检测并提取 Thought（思考过程）
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 步骤 3: 检测模型是否输出 Final Answer，如果是则直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)

            # 步骤 4: 检测并提取 Action（要执行的动作）
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            # 解析动作，提取工具名称和参数
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")
            # 安全机制：只有终端命令才需要询问用户确认，其他工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            # 步骤 5: 执行工具并获取观察结果
            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            # 将观察结果添加到消息历史中，供下一轮循环使用
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})


    def get_tool_list(self) -> str:
        """
        生成工具列表字符串，包含函数签名和简要说明
        
        这个方法用于生成工具描述，供系统提示词使用，让模型知道有哪些工具可用。
        
        Returns:
            格式化的工具列表字符串，每行包含工具名称、签名和文档字符串
        """
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__  # 函数名称
            signature = str(inspect.signature(func))  # 函数签名（参数列表）
            doc = inspect.getdoc(func)  # 函数的文档字符串（说明）
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """
        渲染系统提示模板，替换模板变量
        
        将系统提示词模板中的占位符替换为实际值，包括：
        - 操作系统名称
        - 可用工具列表
        - 项目目录中的文件列表
        
        Args:
            system_prompt_template: 包含占位符的系统提示词模板
            
        Returns:
            替换变量后的完整系统提示词
        """
        tool_list = self.get_tool_list()  # 获取工具列表描述
        # 获取项目目录中所有文件的绝对路径列表
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        # 使用 Template 替换模板变量
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            tool_list=tool_list,
            file_list=file_list
        )

    @staticmethod
    def get_api_key() -> str:
        """
        从环境变量中加载 API 密钥
        
        从 .env 文件中读取 OPENROUTER_API_KEY 环境变量。
        如果未找到，会抛出异常。
        
        Returns:
            API 密钥字符串
            
        Raises:
            ValueError: 如果未找到 API 密钥
        """
        load_dotenv()  # 加载 .env 文件中的环境变量
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("未找到 OPENROUTER_API_KEY 环境变量，请在 .env 文件中设置。")
        return api_key

    def call_model(self, messages):
        """
        调用 LLM 模型生成响应
        
        向 OpenAI API 发送消息列表，获取模型的回复。
        同时将模型的回复添加到消息历史中，用于后续的对话上下文。
        
        Args:
            messages: 消息列表，包含对话历史
            
        Returns:
            模型生成的文本内容
        """
        print("\n\n正在请求模型，请稍等...")
        # 调用 OpenAI API 生成回复
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        # 将模型的回复添加到消息历史中，保持对话上下文
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        """
        解析动作字符串，提取函数名和参数列表
        
        从类似 "function_name(arg1, arg2, ...)" 的字符串中提取：
        - 函数名称
        - 参数列表（支持字符串、数字等多种类型）
        
        这个方法需要特别处理：
        - 多行字符串参数
        - 嵌套括号
        - 转义字符
        
        Args:
            code_str: 动作字符串，格式为 "function_name(arg1, arg2, ...)"
            
        Returns:
            元组 (函数名, 参数列表)
            
        Raises:
            ValueError: 如果动作字符串格式无效
        """
        # 使用正则表达式匹配函数调用格式
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)  # 提取函数名
        args_str = match.group(2).strip()  # 提取参数部分

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""  # 当前正在解析的参数
        in_string = False  # 是否在字符串内部
        string_char = None  # 字符串的引号类型（' 或 "）
        i = 0
        paren_depth = 0  # 括号嵌套深度，用于处理嵌套函数调用
        
        while i < len(args_str):
            char = args_str[i]
            
            if not in_string:
                # 不在字符串内部时
                if char in ['"', "'"]:
                    # 遇到引号，进入字符串模式
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    # 遇到左括号，增加嵌套深度
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    # 遇到右括号，减少嵌套深度
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                # 在字符串内部时，直接添加字符
                current_arg += char
                # 检查是否是字符串结束（考虑转义字符）
                if char == string_char and (i == 0 or args_str[i-1] != '\\'):
                    in_string = False
                    string_char = None
            
            i += 1
        
        # 添加最后一个参数（如果存在）
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))
        
        return func_name, args
    
    def _parse_single_arg(self, arg_str: str):
        """
        解析单个参数，支持多种数据类型
        
        将字符串形式的参数转换为实际的 Python 对象：
        - 字符串字面量（带引号的字符串）
        - 数字（整数、浮点数）
        - 布尔值
        - None
        - 列表、字典等（通过 ast.literal_eval）
        
        Args:
            arg_str: 参数字符串
            
        Returns:
            解析后的参数值（可能是字符串、数字、布尔值等）
        """
        arg_str = arg_str.strip()
        
        # 如果是字符串字面量（被引号包围）
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
           (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符：\" \' \n \t \r \\
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str
        
        # 尝试使用 ast.literal_eval 安全地解析其他类型（数字、布尔值、列表等）
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        """
        获取操作系统名称（用户友好的格式）
        
        将 platform.system() 返回的系统标识符转换为更易读的名称。
        
        Returns:
            操作系统名称字符串（"macOS", "Windows", "Linux" 或 "Unknown"）
        """
        # 系统标识符到友好名称的映射
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")


def read_file(file_path):
    """
    读取文件内容
    
    工具函数：Agent 可以使用此函数读取指定路径的文件内容。
    
    Args:
        file_path: 要读取的文件路径
        
    Returns:
        文件内容的字符串
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_height(name):
    """
    获取某个建筑的高度
    
    Args:
        name: 建筑名称
    """
    return f"{name} 的高度为 100 米"

def write_to_file(file_path, content):
    """
    将指定内容写入指定文件
    
    工具函数：Agent 可以使用此函数将内容写入文件。
    会自动将字符串中的 "\\n" 转换为实际的换行符。
    
    Args:
        file_path: 要写入的文件路径
        content: 要写入的内容
        
    Returns:
        "写入成功" 字符串
    """
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"

def run_terminal_command(command):
    """
    执行终端命令
    
    工具函数：Agent 可以使用此函数执行系统终端命令。
    这是一个危险操作，需要用户确认。
    
    Args:
        command: 要执行的终端命令字符串
        
    Returns:
        如果执行成功返回 "执行成功"，否则返回错误信息
    """
    import subprocess
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr

@click.command()
@click.argument('project_directory',
                type=click.Path(exists=True, file_okay=False, dir_okay=True))
def main(project_directory):
    """
    主函数：启动 ReAct Agent
    
    程序入口点，初始化 Agent 并执行用户任务。
    
    Args:
        project_directory: 项目目录路径（命令行参数）
    """
    # 获取项目目录的绝对路径
    project_dir = os.path.abspath(project_directory)

    # 定义 Agent 可用的工具列表
    tools = [read_file, write_to_file, run_terminal_command, get_height]
    
    # 初始化 ReAct Agent
    # 可选模型：
    # - "openai/gpt-4o": OpenAI GPT-4o 模型
    # - "Pro/deepseek-ai/DeepSeek-R1": DeepSeek R1 模型（当前使用）
    #agent = ReActAgent(tools=tools, model="openai/gpt-4o", project_directory=project_dir)
    #model = "Pro/zai-org/GLM-4.7"
    #model="Pro/deepseek-ai/DeepSeek-R1"
    model = "deepseek-ai/DeepSeek-V3.2"
    agent = ReActAgent(tools=tools, model=model, project_directory=project_dir)
    
    # 获取用户输入的任务
    task = input("请输入任务：")

    # 运行 Agent 执行任务
    final_answer = agent.run(task)

    # 输出最终答案
    print(f"\n\n✅ Final Answer：{final_answer}")

if __name__ == "__main__":
    main()


## Optional Task 
# 1. 创建一个网站， 网站内容是关于某个公司的介绍
# 2. 创建一个游戏， 具体的游戏规则设定
# 3. 查询某个建筑的高度

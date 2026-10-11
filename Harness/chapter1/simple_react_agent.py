"""
简单 Agent（整合版）：将 prompt_template.py、react_agent.py、simple_claude_agent.py
合并为一个可直接运行的 Python 文件。

模型使用阿里云千问（Qwen），通过 DashScope OpenAI 兼容接口调用。
运行前请配置环境变量 DASHSCOPE_API_KEY（可放在 .env 文件中）。

用法：
    python simple_react_agent.py [项目目录]
    不传参数时默认使用当前工作目录。
"""

import ast
import inspect
import json
import os
import platform
import re
import subprocess
from string import Template
from typing import List, Callable, Tuple

import httpx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 系统提示词模板（原 prompt_template.py）
# ============================================================
react_system_prompt_template = """
你需要解决一个问题。为此，你需要将问题分解为多个步骤。对于每个步骤，首先使用 <thought> 思考要做什么，然后使用可用工具之一决定一个 <action>。接着，你将根据你的行动从环境/工具中收到一个 <observation>。持续这个思考和行动的过程，直到你有足够的信息来提供 <final_answer>。
所有步骤请严格使用以下 XML 标签格式输出：
- <question> 用户问题
- <thought> 思考
- <action> 采取的工具操作
- <observation> 工具或环境返回的结果
- <final_answer> 最终答案
⸻
例子 1:
<question>埃菲尔铁塔有多高？</question>
<thought>我需要找到埃菲尔铁塔的高度。可以使用搜索工具。</thought>
<action>get_height("埃菲尔铁塔")</action>
<observation>埃菲尔铁塔的高度约为330米（包含天线）。</observation>
<thought>搜索结果显示了高度。我已经得到答案了。</thought>
<final_answer>埃菲尔铁塔的高度约为330米。</final_answer>
⸻
请严格遵守：
- 你每次回答都必须包括两个标签，第一个是 <thought>，第二个是 <action> 或 <final_answer>
- 输出 <action> 后立即停止生成，等待真实的 <observation>，擅自生成 <observation> 将导致错误
- 如果 <action> 中的某个工具参数有多行的话，请使用 \n 来表示，如：<action>write_to_file("/tmp/test.txt", "a\nb\nc")</action>
- 工具参数中的文件路径请使用绝对路径，不要只给出一个文件名。比如要写 write_to_file("/tmp/test.txt", "内容")，而不是 write_to_file("test.txt", "内容")
⸻
本次任务可用工具：
${tool_list}
⸻
环境信息：
操作系统：${operating_system}
工作目录：${working_directory}
工作目录下文件列表：${file_list}
"""


# ============================================================
# 工具函数（原 simple_claude_agent.py）
# ============================================================
def read_file(file_path):
    """用于读取文件内容"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def write_to_file(file_path, content):
    """将指定内容写入指定文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.replace("\\n", "\n"))
    return "写入成功"


def run_terminal_command(command):
    """用于执行终端命令"""
    run_result = subprocess.run(command, shell=True, capture_output=True, text=True)
    return "执行成功" if run_result.returncode == 0 else run_result.stderr


# ============================================================
# ReAct Agent（原 react_agent.py，模型改为阿里云千问）
# ============================================================
class ReActAgent:
    def __init__(self, tools: List[Callable], model: str, project_directory: str):
        self.tools = {func.__name__: func for func in tools}
        self.model = model
        self.project_directory = project_directory
        # 阿里云百炼（DashScope）OpenAI 兼容接口
        self.client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            http_client=httpx.Client(
                timeout=httpx.Timeout(60.0, connect=10.0),
                verify=False,
                trust_env=False,  # 避免系统代理导致连接超时
            ),
        )

    # 打印发送给AI的请求内容
    def print_messages(self, messages, description: str):
        print("\n" + "=" * 50)
        print(description)
        print("=" * 50)
        # 打印整个messages的JSON内容
        print(json.dumps(messages, indent=4, ensure_ascii=False))
        print("=" * 50)

    def run(self, user_input: str):
        messages = [
            {"role": "system", "content": self.render_system_prompt(react_system_prompt_template)},
            {"role": "user", "content": f"<question>{user_input}</question>"}
        ]

        max_iterations = 10  # 防止无限循环
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            # 打印请求
            self.print_messages(messages, f"第 {iteration} 次API调用, 请求内容：")

            # 请求模型
            content = self.call_model(messages)

            # 打印返回
            self.print_messages(content, f"第 {iteration} 次API调用, 返回内容：")

            # 检测 Thought
            thought_match = re.search(r"<thought>(.*?)</thought>", content, re.DOTALL)
            if thought_match:
                thought = thought_match.group(1)
                print(f"\n\n💭 Thought: {thought}")

            # 检测模型是否输出 Final Answer，如果是的话，直接返回
            if "<final_answer>" in content:
                final_answer = re.search(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                return final_answer.group(1)

            # 检测 Action
            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")
            action = action_match.group(1)
            tool_name, args = self.parse_action(action)

            print(f"\n\n🔧 Action: {tool_name}({', '.join(args)})")
            # 只有终端命令才需要询问用户，其他的工具直接执行
            should_continue = input(f"\n\n是否继续？（Y/N）") if tool_name == "run_terminal_command" else "y"
            if should_continue.lower() != 'y':
                print("\n\n操作已取消。")
                return "操作被用户取消"

            try:
                observation = self.tools[tool_name](*args)
            except Exception as e:
                observation = f"工具执行错误：{str(e)}"
            print(f"\n\n🔍 Observation：{observation}")
            obs_msg = f"<observation>{observation}</observation>"
            messages.append({"role": "user", "content": obs_msg})

    def get_tool_list(self) -> str:
        """生成工具列表字符串，包含函数签名和简要说明"""
        tool_descriptions = []
        for func in self.tools.values():
            name = func.__name__
            signature = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            tool_descriptions.append(f"- {name}{signature}: {doc}")
        return "\n".join(tool_descriptions)

    def render_system_prompt(self, system_prompt_template: str) -> str:
        """渲染系统提示模板，替换变量"""
        tool_list = self.get_tool_list()
        file_list = ", ".join(
            os.path.abspath(os.path.join(self.project_directory, f))
            for f in os.listdir(self.project_directory)
        )
        return Template(system_prompt_template).substitute(
            operating_system=self.get_operating_system_name(),
            working_directory=self.project_directory,
            tool_list=tool_list,
            file_list=file_list
        )

    def call_model(self, messages):
        print("\n\n正在请求模型，请稍等...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = response.choices[0].message.content
        messages.append({"role": "assistant", "content": content})
        return content

    def parse_action(self, code_str: str) -> Tuple[str, List[str]]:
        match = re.match(r'(\w+)\((.*)\)', code_str, re.DOTALL)
        if not match:
            raise ValueError("Invalid function call syntax")

        func_name = match.group(1)
        args_str = match.group(2).strip()

        # 手动解析参数，特别处理包含多行内容的字符串
        args = []
        current_arg = ""
        in_string = False
        string_char = None
        i = 0
        paren_depth = 0

        while i < len(args_str):
            char = args_str[i]

            if not in_string:
                if char in ['"', "'"]:
                    in_string = True
                    string_char = char
                    current_arg += char
                elif char == '(':
                    paren_depth += 1
                    current_arg += char
                elif char == ')':
                    paren_depth -= 1
                    current_arg += char
                elif char == ',' and paren_depth == 0:
                    # 遇到顶层逗号，结束当前参数
                    args.append(self._parse_single_arg(current_arg.strip()))
                    current_arg = ""
                else:
                    current_arg += char
            else:
                current_arg += char
                if char == string_char and (i == 0 or args_str[i - 1] != '\\'):
                    in_string = False
                    string_char = None

            i += 1

        # 添加最后一个参数
        if current_arg.strip():
            args.append(self._parse_single_arg(current_arg.strip()))

        return func_name, args

    def _parse_single_arg(self, arg_str: str):
        """解析单个参数"""
        arg_str = arg_str.strip()

        # 如果是字符串字面量
        if (arg_str.startswith('"') and arg_str.endswith('"')) or \
                (arg_str.startswith("'") and arg_str.endswith("'")):
            # 移除外层引号并处理转义字符
            inner_str = arg_str[1:-1]
            # 处理常见的转义字符
            inner_str = inner_str.replace('\\"', '"').replace("\\'", "'")
            inner_str = inner_str.replace('\\n', '\n').replace('\\t', '\t')
            inner_str = inner_str.replace('\\r', '\r').replace('\\\\', '\\')
            return inner_str

        # 尝试使用 ast.literal_eval 解析其他类型
        try:
            return ast.literal_eval(arg_str)
        except (SyntaxError, ValueError):
            # 如果解析失败，返回原始字符串
            return arg_str

    def get_operating_system_name(self):
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux"
        }

        return os_map.get(platform.system(), "Unknown")


# ============================================================
# 程序入口
# ============================================================
def main(project_directory: str = "."):
    # 确保项目目录存在，如果不存在则创建
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)

    project_dir = os.path.abspath(project_directory)

    tools = [read_file, write_to_file, run_terminal_command]
    # 阿里云千问模型
    agent = ReActAgent(tools=tools, model="qwen-plus", project_directory=project_dir)

    task = input("请输入任务：")

    final_answer = agent.run(task)

    print(f"\n\n✅ Final Answer：{final_answer}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

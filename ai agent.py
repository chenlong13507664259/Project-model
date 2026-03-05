import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
from pathlib import Path
from datetime import datetime
import gradio as gr

# 设置环境变量优化 CPU 性能
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'


class QwenAgent:
    def __init__(self, model_path=None):
        """
        初始化 Qwen 模型

        Args:
            model_path: 模型路径，可以是 Hugging Face 模型名称或本地路径
        """
        # 使用 Qwen2.5-1.5B-Instruct 模型
        if model_path is None:
            model_path = "Qwen/Qwen2.5-1.5B-Instruct"

        print(f"正在加载模型：{model_path} ...")

        # 检查是否是本地路径
        if os.path.exists(model_path):
            print(f"检测到本地模型路径：{model_path}")
            local_model = True
        else:
            # 检查 Hugging Face 缓存目录 - 跨平台兼容
            cache_dir = Path.home() / ".cache" / "huggingface"
            print(f"Hugging Face 缓存目录：{cache_dir}")

            # 检查缓存中是否有模型
            model_cache_exists = False
            if cache_dir.exists():
                # 在缓存目录中查找 Qwen2.5-1.5B 模型
                for model_dir in cache_dir.glob("**/models--*"):
                    model_path_str = str(model_dir)
                    if "Qwen" in model_path_str and "1.5B" in model_path_str:
                        print(f"找到缓存的模型：{model_dir}")
                        model_cache_exists = True
                        break

            if model_cache_exists:
                print("✅ 使用缓存的模型，无需重新下载")
                local_model = False
            else:
                print("❌ 未找到缓存的模型，将从 Hugging Face 下载...")
                print("💡 下载完成后会自动保存，下次运行无需重新下载")
                print(f"📦 模型大小约 3-4GB，请耐心等待...\n")
                local_model = False

        # 加载 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side='left',
            local_files_only=False,
            cache_dir=str(cache_dir) if not local_model else None
        )

        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # CPU 运行
        self.device = torch.device("cpu")
        print(f"使用设备：{self.device}")

        # 加载模型 - CPU 使用 float32
        print("正在加载模型到内存...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            local_files_only=False,
            cache_dir=str(cache_dir) if not local_model else None
        )

        print("✅ 模型加载完成！")

    def chat(self, message, history=None, max_new_tokens=256, temperature=0.7, top_p=0.9):
        """
        与模型进行对话

        Args:
            message: 用户输入的消息
            history: 对话历史，格式为 [(user_msg, assistant_msg), ...]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数，控制随机性
            top_p: Top-p 采样参数

        Returns:
            模型回复的消息
        """
        if history is None:
            history = []

        # 获取当前时间信息
        current_time = datetime.now()
        current_date_str = current_time.strftime("%Y年%m月%d日 %H:%M")
        weekday_map = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日"
        }
        weekday_str = weekday_map[current_time.weekday()]

        # 检测用户问题中是否包含特定日期
        import re
        date_pattern = r'(\d{4}) 年 (\d{1,2}) 月 (\d{1,2}) [日号]'
        match = re.search(date_pattern, message)

        extra_info = ""
        if match:
            # 如果用户问了特定日期，用 Python 计算星期几
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            try:
                target_date = datetime(year, month, day)
                target_weekday = weekday_map[target_date.weekday()]
                extra_info = f"\n\n重要：你被问到关于{year}年{month}月{day}日的问题，这一天是{target_weekday}。请在回答中使用这个准确信息。"
            except ValueError:
                pass

        # 判断用户是否在询问时间相关问题
        time_keywords = ['时间', '日期', '星期', '几点', '今天', '明天', '昨天', '几号', '哪年', '哪月', '哪天']
        is_time_question = any(keyword in message for keyword in time_keywords) or match

        # 构建系统消息 - 只在需要时提供时间信息
        if is_time_question:
            system_content = f"""你是一个智能助手。当前准确时间是{current_date_str}，{weekday_str}。

规则：
1. 当用户询问日期、时间、星期几时，必须使用上述时间信息回答
2. 不要说你无法获取实时信息，你已经知道当前时间
3. 直接基于给定的时间信息回答，不要编造其他时间{extra_info}

现在请回答用户的问题。"""
        else:
            system_content = "你是一个智能助手，请用友好、简洁的语气回答用户的问题。"

        # 构建对话消息
        messages = [{"role": "system", "content": system_content}]

        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": message})

        # 应用聊天模板
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 分词
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # CPU 优化的生成配置
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True if temperature > 0 else False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            num_beams=1,
            early_stopping=False,
        )

        # 生成回复
        generated_ids = self.model.generate(
            **model_inputs,
            generation_config=generation_config
        )

        # 提取生成的内容
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response


def create_gradio_ui(agent):
    """创建 Gradio 界面"""

    def predict(message, history):
        """Gradio 预测函数"""
        if not message:
            return "", []

        try:
            # 转换历史格式为 chat 方法需要的格式 [(user_msg, ai_msg), ...]
            chat_history = []
            if history:
                for msg in history:
                    if isinstance(msg, dict):
                        if msg['role'] == 'user':
                            last_user_msg = msg['content']
                        elif msg['role'] == 'assistant' and len(chat_history) > 0:
                            chat_history.append((last_user_msg, msg['content']))

            # 获取回复
            response = agent.chat(message, history=chat_history, max_new_tokens=512)

            # 构建新的历史记录（messages 格式）
            if history is None:
                new_history = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ]
            else:
                new_history = list(history) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": response}
                ]

            return "", new_history
        except Exception as e:
            error_msg = f"发生错误：{e}"
            if history is None:
                new_history = [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_msg}
                ]
            else:
                new_history = list(history) + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": error_msg}
                ]
            return "", new_history

    # 创建 Gradio 界面
    with gr.Blocks(title="Qwen AI 助手") as demo:
        gr.Markdown("""
        # 🤖 Qwen AI 助手

        基于 Qwen2.5-1.5B-Instruct 模型的智能聊天机器人

           ✅ 多轮对话记忆

        """)

        chatbot = gr.Chatbot(
            label="AI 助手",
            height=500
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="输入消息...（按 Enter 发送）",
                scale=8,
                show_label=False,
                container=False
            )
            send_btn = gr.Button("发送", scale=1, variant="primary")

        with gr.Row():
            clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
            reload_btn = gr.Button("🔄 重新加载", variant="secondary")

        # 示例问题
        gr.Examples(
            examples=[
                "你好！",
                "今天星期几？",
                "现在几点了？",
                "帮我写一首诗",
                "什么是人工智能？"
            ],
            inputs=msg_input
        )

        # 绑定事件 - 使用正确的格式
        msg_input.submit(
            fn=predict,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        send_btn.click(
            fn=predict,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )

        clear_btn.click(
            fn=lambda: ("", []),
            inputs=None,
            outputs=[msg_input, chatbot]
        )

        reload_btn.click(
            fn=lambda: None,
            inputs=None,
            outputs=None,
            queue=False,
            js="() => location.reload()"
        )

        gr.Markdown("""
        ---
        ⚠️ 注意：CPU 运行速度较慢，请耐心等待回复
        """)

    return demo


def main():
    """主函数"""
    print("=" * 60)
    print("Qwen2.5-1.5B-Instruct CPU 聊天机器人")
    print("=" * 60)
    print("\n提示：")
    print("- CPU 运行较慢，请耐心等待")
    print("- 首次运行会下载模型（约 3-4GB），之后会自动使用缓存")
    print(f"- 缓存位置：{Path.home() / '.cache' / 'huggingface'}\n")

    # 初始化模型
    try:
        agent = QwenAgent()
    except Exception as e:
        print(f"模型加载失败：{e}")
        print("\n请确保:")
        print("1. 已安装所有依赖包 (pip install -r requirements.txt)")
        print("2. 有足够的内存（至少 8GB）")
        print("3. 网络连接正常（首次需要从 Hugging Face 下载模型）")
        return

    print("\n✅ 模型加载成功！")

    # Web UI 模式
    print("\n" + "=" * 60)
    print("启动 Web UI 界面...")
    print("=" * 60)
    print("\n💡 界面将在浏览器中自动打开")
    print("如果未自动打开，请访问：http://localhost:7861\n")

    demo = create_gradio_ui(agent)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(),
        inbrowser=True
    )


if __name__ == "__main__":
    main()

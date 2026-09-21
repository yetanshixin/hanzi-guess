import os
import json
from typing import Any
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime
from openai import OpenAI

app = FastAPI(title="汉字谜盒")
app.mount("/static", StaticFiles(directory="static"), name="static")

if not os.path.exists("sessions"):
    os.mkdir("sessions")

SYSTEM_PROMPT = """
# 角色定义
你是一个专门玩猜字谜的AI小助手，只进行字谜互动，不闲聊无关内容，全程纯文本交互，不使用表情符号。

## 核心能力
- 出字谜、判对错、给提示
- 记忆已用谜题，确保会话内不重复
- 简洁明快回应

## 出题规则（严格执行！）
1. 开场先友好打招呼，并随机出一道常见、简单、适合大众并必须符合逻辑推理的字谜，禁止使用生僻、低俗、网络烂梗。
2. 题目格式：“谜面”（打一字）。
3. 每次出题必须完全随机，禁止重复使用相同题目，也可以偶尔穿插使用，下面示例中的谜语。
4. 新出题目时, 不要提示, 用户需要提示时, 或者答错时, 再给予合理的提示。

## 判题规则（严格执行！）
1. 用户只回复一个字时，直接视为答案。
2. 答对：立即夸奖并揭晓谜底，格式如“太棒了！就是‘X’字！要不要再来一题？”
3. 答错：告知不对，可给一句简短提示，但不泄露答案。格式如“不对哦，再想想~”
4. 严禁在用户答错后直接公布答案！只有用户说“公布答案”或“不知道”等情况时才公布。

## 互动流程
1. 用户答对：夸奖 + 确认正确 + 询问“要不要再来一题？”
2. 用户答错：告知不对 + 简单提示 + 鼓励继续猜
3. 用户说“提示一下”：给出简短线索，不公布答案
4. 用户说“公布答案”或“不知道”：揭晓谜底并解释 + 询问“要不要再来一题？”
5. 用户说“换一题”“再来一题”：立即更换新字谜

## 回复风格约束
- 语气轻松有趣，但保持简洁
- 全程只围绕字谜，拒绝回答其他问题
- 回复不超过3句话
- **绝对不要在回复中说“这个出过了，我来个新的”或类似表述** — 直接给出新谜语即可
- 判题错误零容忍，不确定谜底时，先回复“我再想想”而不是乱判

## 常见谜语类型及谜底参考示例, 仅仅为参照示例
### 组合类
- 「一加一不是二」= 王
- 「二人不是天」= 夫
- 「十口不是田」= 古

### 包含类
- 「一人在内」= 肉
- 「口里有人」= 囚
- 「门里有口」= 问
- 「田里长草」= 苗
- 「心里有你」= 您
- 「山里有山」= 出
- 「王头上有人」= 全
- 「水上有石」= 泵

### 半取类
- 「半吃半拿」= 哈
- 「半真半假」= 值
- 「半青半紫」= 素
- 「半朋半友」= 有
- 「半推半就」= 扰
- 「半山半水」= 汕

### 象形类
- 「三人又重逢」= 众
- 「一口咬掉牛尾巴」= 告
- 「两座山」= 出
- 「三日又重逢」= 晶
"""

client = OpenAI(
    api_key=os.environ.get("MY_SECRET_KEY"),
    base_url="https://api.deepseek.com"
)


def get_session_file_name(session_id):
    return f"sessions/{session_id}.json"


def generate_session_id():
    return datetime.now().strftime("%Y%m%d%H%M%S")


class APIResponse(BaseModel):
    code: int
    message: str
    data: Any


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.exception_handler(Exception)
def handle_exception(request: Request, exc: Exception):
    return JSONResponse(content={"code": 500, "message": f"url:{request.url} err:{exc}", "data": None})


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return FileResponse("static/index.html")


@app.post("/api/sessions")
def creat_session():
    session_id = generate_session_id()

    session_data = {
        "current_session": session_id,
        "messages": [],
    }
    with open(get_session_file_name(session_id), "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return APIResponse(code=200, message="success", data=session_id)


@app.post("/api/chat")
def chat(request: ChatRequest):
    session_path = get_session_file_name(request.session_id)
    with open(session_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in session_data["messages"]:
        messages.append(message)
    messages.append({"role": "user", "content": request.message})

    api_kwargs = {
        "model": "deepseek-flash",
        "messages": messages,
        "stream": False,
        "extra_body": {"thinking": {"type": "disabled"}},
        "reasoning_effort": "low"
    }
    response = client.chat.completions.create(**api_kwargs)
    ai_response = response.choices[0].message.content

    messages.append({"role": "assistant", "content": ai_response})
    session_data["messages"] = messages[1::]
    with open(session_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)

    return APIResponse(code=200, message="success", data=ai_response)


@app.api_route("/api/sessions", methods=["GET", "HEAD"])
def get_sessions():
    session_ids = [file.split(".")[0] for file in os.listdir("sessions")]
    session_ids.sort(reverse=True)
    return APIResponse(code=200, message="success", data=session_ids)


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    session_file = get_session_file_name(session_id)
    with open(session_file, "r", encoding="utf-8") as f:
        session_data = json.load(f)
    return APIResponse(code=200, message="success", data=session_data)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    session_file = get_session_file_name(session_id)
    if os.path.exists(session_file):
        os.remove(session_file)
    return APIResponse(code=200, message="success", data=None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8000, access_log=False)

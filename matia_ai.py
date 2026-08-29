import os
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

# ============================================================
# MATIA AI
# One-file Flask + Groq AI
# ============================================================

app = Flask(__name__)

# ------------------------------------------------------------
# GROQ
# ------------------------------------------------------------

API_KEY = os.environ.get("GROQ_API_KEY", "").strip()

MODEL = os.environ.get(
    "MATIA_MODEL",
    "openai/gpt-oss-120b"
)

client = Groq(api_key=API_KEY) if API_KEY else None

# ------------------------------------------------------------
# CONVERSATION MEMORY
# ------------------------------------------------------------

conversations = {}

MAX_HISTORY = 12
MAX_MESSAGE_LENGTH = 12000
MAX_OUTPUT_TOKENS = 5000

# ------------------------------------------------------------
# 50 EXPERTS
# ------------------------------------------------------------

EXPERTS = [
    "Coding & Programming",
    "Reasoning & Problem Solving",
    "Mathematics",
    "Studying & Tutoring",
    "Science",
    "Web Research",
    "Documents",
    "Files & Projects",
    "Writing",
    "Editing & Proofreading",
    "Translation",
    "Language Learning",
    "Creative Ideas",
    "Image Understanding",
    "Video Concepts",
    "Music & Lyrics",
    "Game Development",
    "Roblox & Luau",
    "Web Development",
    "App Development",
    "AI Development",
    "APIs & Integrations",
    "Databases",
    "Cloud & Deployment",
    "Git & GitHub",
    "Linux",
    "Windows",
    "DevOps",
    "Docker",
    "Debugging",
    "Testing & Quality",
    "Defensive Cybersecurity",
    "Privacy & Security",
    "Data Analysis",
    "Statistics",
    "Logic",
    "Puzzles",
    "Organization",
    "Planning",
    "Goal Planning",
    "Brainstorming",
    "Checklists",
    "Summarization",
    "Information Extraction",
    "Reports",
    "Teacher Mode",
    "Quiz Generator",
    "Flashcards",
    "Exam Preparation",
    "AI Agent & Automation"
]

# ------------------------------------------------------------
# EXPERT KEYWORDS
# ------------------------------------------------------------

KEYWORDS = {
    "Coding & Programming": [
        "code", "coding", "script", "program",
        "python", "javascript", "typescript",
        "html", "css", "java", "c++", "c#",
        "php", "ruby", "rust", "golang",
        "lua", "luau", "sql", "bash"
    ],

    "Mathematics": [
        "math", "mathematics", "calculate",
        "equation", "algebra", "geometry",
        "percentage", "fraction", "probability",
        "calculus", "integral", "derivative"
    ],

    "Studying & Tutoring": [
        "study", "learn", "school", "homework",
        "lesson", "teach", "explain", "exam",
        "revision", "test"
    ],

    "Writing": [
        "write", "rewrite", "essay", "email",
        "message", "story", "caption",
        "paragraph", "post"
    ],

    "Translation": [
        "translate", "translation", "english",
        "albanian", "italian", "spanish",
        "german", "french"
    ],

    "Roblox & Luau": [
        "roblox", "roblox studio", "luau",
        "localscript", "serverscript",
        "leaderstats", "remoteevent"
    ],

    "Game Development": [
        "game", "unity", "godot", "unreal",
        "npc", "inventory", "quest",
        "level", "shop system"
    ],

    "Debugging": [
        "error", "bug", "debug", "broken",
        "fix", "not working", "traceback",
        "syntaxerror", "exception"
    ],

    "AI Development": [
        "ai", "artificial intelligence",
        "machine learning", "llm", "model",
        "prompt", "agent", "groq", "openai"
    ],

    "Data Analysis": [
        "data", "dataset", "csv", "analysis",
        "chart", "graph", "average",
        "median", "statistics"
    ],

    "Planning": [
        "plan", "planning", "roadmap",
        "schedule", "organize", "steps"
    ],

    "Quiz Generator": [
        "quiz", "questions", "test me"
    ],

    "Flashcards": [
        "flashcards", "flash card", "cards"
    ]
}

# ------------------------------------------------------------
# CODE DETECTION
# ------------------------------------------------------------

LANGUAGES = {
    "Python": ["python", ".py"],
    "JavaScript": ["javascript", ".js"],
    "TypeScript": ["typescript", ".ts"],
    "HTML": ["html", ".html"],
    "CSS": ["css", ".css"],
    "Lua/Luau": ["lua", "luau", ".lua"],
    "Java": ["java", ".java"],
    "C++": ["c++", ".cpp"],
    "C#": ["c#", "csharp", ".cs"],
    "PHP": ["php", ".php"],
    "Ruby": ["ruby", ".rb"],
    "Rust": ["rust", ".rs"],
    "Go": ["golang", ".go"],
    "SQL": ["sql", ".sql"],
    "Bash": ["bash", ".sh"],
    "PowerShell": ["powershell", ".ps1"]
}


def detect_language(text):
    text = text.lower()

    for language, words in LANGUAGES.items():
        for word in words:
            if word in text:
                return language

    return "Auto"


def is_code_request(text):
    text = text.lower()

    if "```" in text:
        return True

    actions = [
        "make", "build", "create",
        "write", "generate", "code",
        "script", "program", "implement",
        "fix", "debug", "website",
        "calculator", "app"
    ]

    has_language = any(
        word in text
        for values in LANGUAGES.values()
        for word in values
    )

    has_action = any(
        action in text
        for action in actions
    )

    return has_language and has_action


# ------------------------------------------------------------
# AUTO EXPERT DETECTION
# ------------------------------------------------------------

def detect_experts(text):
    text = text.lower()
    result = []

    if is_code_request(text):
        result.append("Coding & Programming")

    for expert, keywords in KEYWORDS.items():

        if any(keyword in text for keyword in keywords):

            if expert not in result:
                result.append(expert)

    if not result:
        result.append("Reasoning & Problem Solving")

    return result[:7]


# ------------------------------------------------------------
# SYSTEM PROMPT
# ------------------------------------------------------------

SYSTEM_PROMPT = f"""
You are MATIA AI, an advanced general-purpose AI assistant.

You have 50 expert areas:

{chr(10).join("- " + expert for expert in EXPERTS)}

Your job is to automatically understand the user's request
and select the appropriate expert or combination of experts.

IMPORTANT GENERAL RULES:

1. Understand the actual request.
2. Use previous conversation context.
3. Do not ask unnecessary questions.
4. If the request is clear, do the task.
5. Be accurate and practical.
6. Answer in the user's language when possible.
7. Do not invent facts.
8. Do not pretend to have tools you don't have.
9. Keep explanations useful rather than repetitive.

CODING MODE:

When the user asks for code:

- ACTUALLY WRITE THE CODE.
- Do not merely explain how to write it.
- Make it copy-paste ready.
- Make it complete and functional.
- Use the requested programming language.
- If they ask for HTML, produce actual HTML.
- If they ask for Python, produce actual Python.
- If they ask for Roblox, use Luau.
- Include necessary CSS/JS when creating a complete website.
- Do not intentionally make code tiny or fake.
- Do not replace the requested code with questions.
- If fixing code, provide the corrected code.
- If the user asks for a full script, provide the full script.

CODE FORMAT:

Use Markdown code blocks:

```python
print("Hello")

Do not put the code in normal conversational text.

MATH MODE:

Solve mathematical problems carefully.
Show useful steps.
Verify calculations.

STUDY MODE:

Teach clearly.
Use examples.
Increase difficulty gradually.
Create quizzes or flashcards when requested.

DEBUG MODE:

When given an error:

1. Identify the likely cause.
2. Explain it briefly.
3. Give the fix.
4. Give complete corrected code when appropriate.

SMART CONTEXT:

Use previous messages to understand references such as:

"make it bigger"
"fix that"
"continue"
"make it Python"
"add this"
"change the UI"

Do not forget the current project requirements.

SAFETY:

For cybersecurity, stay within defensive,
educational and authorized use.
Do not assist with harmful wrongdoing.
"""

------------------------------------------------------------

HISTORY

------------------------------------------------------------

def get_history(conversation_id):

if conversation_id not in conversations:
    conversations[conversation_id] = []

return conversations[conversation_id]

def add_history(conversation_id, role, content):

history = get_history(conversation_id)

history.append({
    "role": role,
    "content": content
})

if len(history) > MAX_HISTORY:
    del history[:-MAX_HISTORY]

------------------------------------------------------------

ASK GROQ

------------------------------------------------------------

def ask_ai(conversation_id, user_message):

if client is None:
    raise RuntimeError(
        "GROQ_API_KEY is missing. "
        "Add GROQ_API_KEY in Render Environment Variables."
    )

history = get_history(conversation_id)

experts = detect_experts(user_message)
language = detect_language(user_message)
code_mode = is_code_request(user_message)

routing = f"""

CURRENT REQUEST:

Detected experts:
{", ".join(experts)}

Detected language:
{language}

Code mode:
{code_mode}
"""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT + routing
    }
]

for item in history[-MAX_HISTORY:]:
    messages.append({
        "role": item["role"],
        "content": item["content"]
    })

messages.append({
    "role": "user",
    "content": user_message
})

response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    max_tokens=MAX_OUTPUT_TOKENS,
    temperature=0.35
)

answer = response.choices[0].message.content

return answer, experts, language, code_mode

============================================================

WEB UI

============================================================

HTML = """

<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Matia AI</title><style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #08090d;
    color: #f5f5f7;
    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

.app {
    height: 100vh;
    display: flex;
    flex-direction: column;
}

.header {
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 18px;
    border-bottom: 1px solid #242632;
    background: #0c0d12;
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 19px;
    font-weight: bold;
}

.logoIcon {
    width: 35px;
    height: 35px;
    border-radius: 11px;
    display: flex;
    align-items: center;
    justify-content: center;
    background:
        linear-gradient(
            135deg,
            #705cff,
            #b64dff
        );
}

.status {
    color: #888f9f;
    font-size: 12px;
}

.chat {
    flex: 1;
    overflow-y: auto;
    padding: 25px 12px 120px;
}

.container {
    max-width: 900px;
    margin: auto;
}

.welcome {
    text-align: center;
    padding: 80px 15px 25px;
}

.welcome h1 {
    font-size: 48px;
    margin-bottom: 10px;
}

.gradient {
    background:
        linear-gradient(
            90deg,
            #8d7bff,
            #cf66ff,
            #66caff
        );

    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.welcome p {
    color: #9299a8;
    line-height: 1.6;
}

.chips {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
    margin-top: 25px;
}

.chip {
    background: #12151d;
    border: 1px solid #292d39;
    color: #bfc5d0;
    padding: 9px 13px;
    border-radius: 999px;
    cursor: pointer;
}

.chip:hover {
    background: #1b1f29;
    color: white;
}

.message {
    display: flex;
    margin: 17px 0;
}

.user {
    justify-content: flex-end;
}

.bubble {
    max-width: 92%;
    padding: 13px 15px;
    border-radius: 17px;
    line-height: 1.6;
    word-wrap: break-word;
}

.user .bubble {
    background: #202430;
}

.assistant .bubble {
    background: transparent;
}

.codeBox {
    margin: 14px 0;
    background: #0b0d12;
    border: 1px solid #292d39;
    border-radius: 13px;
    overflow: hidden;
}

.codeHeader {
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 10px;
    background: #11141b;
    color: #9299aa;
    font-size: 12px;
}

.copyBtn {
    border: 0;
    background: #252a36;
    color: white;
    border-radius: 7px;
    padding: 6px 10px;
    cursor: pointer;
}

pre {
    margin: 0;
    padding: 15px;
    overflow-x: auto;
}

textarea {
    width: 100%;
    min-height: 45px;
    max-height: 160px;
    resize: none;
    border: 0;
    outline: 0;
    background: transparent;
    color: white;
    font-size: 15px;
    padding: 10px;
}

.inputArea {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 12px;
    background:
        linear-gradient(
            to top,
            #08090d 70%,
            transparent
        );
}

.composer {
    max-width: 900px;
    margin: auto;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    background: #12151c;
    border: 1px solid #292d38;
    border-radius: 18px;
    padding: 7px;
}

.send {
    width: 45px;
    height: 45px;
    border: 0;
    border-radius: 13px;
    color: white;
    background:
        linear-gradient(
            135deg,
            #705cff,
            #ad50ff
        );
    cursor: pointer;
    font-size: 18px;
}

.send:disabled {
    opacity: .5;
}

@media(max-width:600px) {

    .welcome {
        padding-top: 50px;
    }

    .welcome h1 {
        font-size: 37px;
    }

    .bubble {
        max-width: 96%;
    }
}

</style></head><body><div class="app"><header class="header"><div class="logo">
    <div class="logoIcon">M</div>
    Matia AI
</div>

<div class="status" id="status">
    AI Ready
</div>

</header><main class="chat" id="chat"><div class="container" id="messages"><section class="welcome" id="welcome"><h1>
    <span class="gradient">
        Matia AI
    </span>
</h1>

<p>
    Coding • Math • Study • Research •
    Games • AI • 50 Expert Modes
</p>

<div class="chips">

    <button class="chip"
            onclick="quick('Build me a complete HTML calculator')">
        💻 Build Code
    </button>

    <button class="chip"
            onclick="quick('Explain this math problem step by step')">
        🧮 Math
    </button>

    <button class="chip"
            onclick="quick('Teach me Python from beginner level')">
        📚 Study
    </button>

    <button class="chip"
            onclick="quick('Help me debug my code')">
        🔧 Debug
    </button>

</div>

</section></div></main><div class="inputArea"><form class="composer" id="form"><textarea
    id="input"
    placeholder="Ask Matia anything..."
    autocomplete="off"></textarea><button
class="send"
id="send"
type="submit">
➤
</button>

</form></div></div><script>

const input =
    document.getElementById("input");

const send =
    document.getElementById("send");

const messages =
    document.getElementById("messages");

const chat =
    document.getElementById("chat");

const welcome =
    document.getElementById("welcome");

const status =
    document.getElementById("status");

let conversationId =
    localStorage.getItem(
        "matia_conversation_id"
    );

if (!conversationId) {

    conversationId =
        Date.now().toString() +
        Math.random().toString(36);

    localStorage.setItem(
        "matia_conversation_id",
        conversationId
    );
}

function escapeHTML(text) {

    return text
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
}

function renderAnswer(text) {

    const parts =
        text.split(/```([\\s\\S]*?)```/g);

    let html = "";

    for (let i = 0; i < parts.length; i++) {

        if (i % 2 === 1) {

            let code = parts[i];

            let lines =
                code.split("\\n");

            let language = "code";

            if (
                lines.length > 0 &&
                /^[a-zA-Z0-9_+#-]+$/.test(
                    lines[0].trim()
                )
            ) {

                language =
                    lines.shift().trim();
            }

            code =
                lines.join("\\n").trim();

            html += `
                <div class="codeBox">

                    <div class="codeHeader">

                        <span>
                            ${escapeHTML(language)}
                        </span>

                        <button
                            class="copyBtn"
                            onclick="copyCode(this)">
                            📋 Copy
                        </button>

                    </div>

                    <pre><code>${escapeHTML(code)}</code></pre>            </div>
        `;

    } else {

        let normal =
            escapeHTML(parts[i]);

        normal =
            normal.replace(
                /\\*\\*(.*?)\\*\\*/g,
                "<strong>$1</strong>"
            );

        normal =
            normal.replace(
                /`([^`]+)`/g,
                "<code>$1</code>"
            );

        normal =
            normal.replace(
                /\\n/g,
                "<br>"
            );

        html += normal;
    }
}

return html;

}

function addUser(text) {

welcome.style.display = "none";

const div =
    document.createElement("div");

div.className =
    "message user";

div.innerHTML = `
    <div class="bubble">
        ${escapeHTML(text)}
    </div>
`;

messages.appendChild(div);

scrollBottom();

}

function addAssistant(text) {

const div =
    document.createElement("div");

div.className =
    "message assistant";

const bubble =
    document.createElement("div");

bubble.className =
    "bubble";

bubble.innerHTML =
    renderAnswer(text);

div.appendChild(bubble);

messages.appendChild(div);

scrollBottom();

}

function loading() {

const div =
    document.createElement("div");

div.id = "loading";

div.className =
    "message assistant";

div.innerHTML = `
    <div class="bubble">
        🧠 Matia is thinking...
    </div>
`;

messages.appendChild(div);

scrollBottom();

}

function removeLoading() {

const item =
    document.getElementById("loading");

if (item) {
    item.remove();
}

}

function scrollBottom() {

chat.scrollTop =
    chat.scrollHeight;

}

async function sendMessage(text) {

text = text.trim();

if (!text || send.disabled) {
    return;
}

addUser(text);

input.value = "";

send.disabled = true;

status.textContent =
    "Thinking...";

loading();

try {

    const response =
        await fetch("/chat", {

            method: "POST",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                message: text,
                conversation_id:
                    conversationId
            })
        });

    const data =
        await response.json();

    removeLoading();

    if (!response.ok ||
        data.error) {

        addAssistant(
            "❌ " +
            (data.error ||
             "Something went wrong.")
        );

    } else {

        addAssistant(
            data.answer
        );
    }

} catch (error) {

    removeLoading();

    addAssistant(
        "❌ Connection error."
    );

} finally {

    send.disabled = false;

    status.textContent =
        "AI Ready";

    input.focus();
}

}

function quick(text) {

input.value = text;

sendMessage(text);

}

async function copyCode(button) {

const code =
    button
        .closest(".codeBox")
        .querySelector("code")
        .innerText;

try {

    await navigator.clipboard
        .writeText(code);

    const old =
        button.innerText;

    button.innerText =
        "✓ Copied";

    setTimeout(() => {

        button.innerText = old;

    }, 1200);

} catch {

    button.innerText =
        "Copy failed";
}

}

document
.getElementById("form")
.addEventListener(
"submit",
function(event) {

        event.preventDefault();

        sendMessage(input.value);
    }
);

input.addEventListener(
"keydown",
function(event) {

    if (
        event.key === "Enter" &&
        !event.shiftKey
    ) {

        event.preventDefault();

        sendMessage(input.value);
    }
}

);

</script></body>
</html>
"""============================================================

ROUTES

============================================================

@app.route("/")
def home():
return render_template_string(HTML)

@app.route("/health")
def health():

return jsonify({
    "status": "online",
    "name": "Matia AI",
    "model": MODEL,
    "api_connected": client is not None,
    "experts": len(EXPERTS)
})

@app.route("/chat", methods=["POST"])
def chat():

try:

    data =
        request.get_json(silent=True) or {}

    message =
        str(data.get("message", "")).strip()

    conversation_id =
        str(
            data.get(
                "conversation_id",
                "default"
            )
        )

    if not message:
        return jsonify({
            "error": "Write a message first."
        }), 400

    if len(message) > MAX_MESSAGE_LENGTH:
        return jsonify({
            "error": "Message is too long."
        }), 400

    add_history(
        conversation_id,
        "user",
        message
    )

    answer, experts, language, code_mode = \
        ask_ai(
            conversation_id,
            message
        )

    add_history(
        conversation_id,
        "assistant",
        answer
    )

    return jsonify({
        "answer": answer,
        "experts": experts,
        "language": language,
        "code_mode": code_mode,
        "conversation_id": conversation_id
    })

except Exception as error:

    print(
        "MATIA ERROR:",
        repr(error)
    )

    return jsonify({
        "error": str(error)
    }), 500

============================================================

RENDER START

============================================================

if name == "main":

port = int(
    os.environ.get(
        "PORT",
        5000
    )
)

print("=" * 60)
print("MATIA AI")
print("=" * 60)
print("Model:", MODEL)
print("Experts:", len(EXPERTS))
print(
    "Groq:",
    "CONNECTED"
    if client
    else "NOT CONNECTED"
)
print("Port:", port)
print("=" * 60)

app.run(
    host="0.0.0.0",
    port=port,
    debug=False
)

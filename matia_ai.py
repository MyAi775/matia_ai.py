import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)

# =========================
# CONFIG
# =========================

MODEL = os.getenv("MATIA_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MAX_HISTORY = 8
MAX_OUTPUT_TOKENS = 6000
MAX_MESSAGE_CHARS = 12000

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

conversations = {}


# =========================
# 50+ EXPERTS
# =========================

EXPERTS = [
    "General AI",
    "Coding",
    "Python",
    "JavaScript",
    "TypeScript",
    "HTML",
    "CSS",
    "Java",
    "C",
    "C++",
    "C#",
    "PHP",
    "Rust",
    "Go",
    "SQL",
    "Lua",
    "Luau",
    "Roblox",
    "Game Development",
    "Web Development",
    "App Development",
    "AI",
    "Machine Learning",
    "API",
    "Database",
    "Git",
    "GitHub",
    "DevOps",
    "Debugging",
    "Testing",
    "Math",
    "Statistics",
    "Science",
    "Study",
    "Teacher",
    "Quiz",
    "Flashcards",
    "Writing",
    "Editing",
    "Translation",
    "Research",
    "Summarization",
    "Data Analysis",
    "Logic",
    "Problem Solving",
    "Planning",
    "Brainstorming",
    "Productivity",
    "Security Education",
    "Documentation",
    "Project Architecture"
]


# =========================
# EXPERT DETECTION
# =========================

def detect_experts(text):
    text = text.lower()
    found = []

    groups = {
        "Coding": [
            "code", "coding", "programming", "script",
            "function", "class", "developer"
        ],
        "Python": ["python", ".py"],
        "JavaScript": ["javascript", "js"],
        "HTML": ["html"],
        "CSS": ["css"],
        "Roblox": ["roblox", "roblox studio", "luau"],
        "Math": [
            "math", "calculate", "equation",
            "algebra", "geometry", "percentage"
        ],
        "Study": [
            "study", "school", "homework",
            "lesson", "learn", "exam"
        ],
        "Debugging": [
            "error", "bug", "debug",
            "broken", "not working", "traceback"
        ],
        "AI": [
            "ai", "artificial intelligence",
            "llm", "machine learning"
        ],
        "API": [
            "api", "endpoint", "webhook",
            "json api"
        ],
        "GitHub": [
            "github", "repository", "repo",
            "commit", "branch"
        ],
        "DevOps": [
            "render", "deploy", "deployment",
            "server", "hosting"
        ],
        "Writing": [
            "write", "writing", "essay",
            "email", "story", "paragraph"
        ],
        "Translation": [
            "translate", "translation"
        ],
        "Quiz": [
            "quiz", "test me", "questions"
        ],
        "Planning": [
            "plan", "roadmap", "steps"
        ],
        "Brainstorming": [
            "idea", "ideas", "brainstorm"
        ]
    }

    for expert, keywords in groups.items():
        if any(word in text for word in keywords):
            found.append(expert)

    if not found:
        found.append("General AI")

    return found[:8]


# =========================
# LANGUAGE DETECTION
# =========================

def detect_language(text):
    text = text.lower()

    languages = {
        "Python": ["python", ".py"],
        "JavaScript": ["javascript", ".js"],
        "TypeScript": ["typescript", ".ts"],
        "HTML": ["html", ".html"],
        "CSS": ["css", ".css"],
        "Java": ["java", ".java"],
        "C++": ["c++", "cpp", ".cpp"],
        "C#": ["c#", "csharp", ".cs"],
        "PHP": ["php", ".php"],
        "Rust": ["rust", ".rs"],
        "Go": ["golang", ".go"],
        "SQL": ["sql", ".sql"],
        "Lua": ["lua", ".lua"],
        "Luau": ["luau"],
        "Bash": ["bash", ".sh"],
        "PowerShell": ["powershell", ".ps1"]
    }

    for language, keywords in languages.items():
        if any(word in text for word in keywords):
            return language

    return "Auto"


# =========================
# SYSTEM PROMPT
# =========================

SYSTEM_PROMPT = """
You are MATIA AI.

You are a powerful multi-purpose AI assistant with 50+ expert areas.

Rules:
- Be helpful, clear and accurate.
- Answer in the user's language when appropriate.
- Do not ask unnecessary questions.
- If the request is clear, complete it directly.
- Never pretend you performed an action that you did not perform.

CODING:
When the user asks for code, give actual code.
When the user asks for full code, give a complete copy-paste-ready implementation.
Use Markdown code blocks.

HTML:
If asked to make an HTML website or application,
provide a complete HTML document with CSS and JavaScript when needed.

DEBUGGING:
If the user gives an error:
1. Identify the likely cause.
2. Explain it briefly.
3. Give the exact fix.
4. Give corrected code when useful.

MATH:
Solve carefully and explain important steps.

STUDY:
Teach clearly and use examples.

CONTEXT:
Understand phrases such as:
"that code",
"same one",
"make it better",
"continue",
"add this".

SECURITY:
For cybersecurity topics, stay within legal,
authorized, defensive and educational use.
"""


# =========================
# HISTORY
# =========================

def get_history(conversation_id):
    return conversations.setdefault(conversation_id, [])


def add_history(conversation_id, role, content):
    history = get_history(conversation_id)

    history.append({
        "role": role,
        "content": content
    })

    if len(history) > MAX_HISTORY:
        conversations[conversation_id] = history[-MAX_HISTORY:]


# =========================
# AI REQUEST
# =========================

def ask_ai(conversation_id, user_message):

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Render Environment Variables."
        )

    experts = detect_experts(user_message)
    language = detect_language(user_message)

    routing = f"""
Current expert routing:
{", ".join(experts)}

Detected programming language:
{language}
"""

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + routing
        }
    ]

    for item in get_history(conversation_id):
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

    if not answer:
        answer = "I couldn't generate an answer."

    return answer, experts, language


# =========================
# FRONTEND
# =========================

HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
content="width=device-width, initial-scale=1.0">

<title>Matia AI</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #08090d;
    color: white;
    font-family: Arial, sans-serif;
}

.header {
    height: 64px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 18px;
    background: #0d0f15;
    border-bottom: 1px solid #252936;
}

.brand {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: bold;
    font-size: 19px;
}

.logo {
    width: 38px;
    height: 38px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #705cff, #c64dff);
}

.status {
    color: #8f96a5;
    font-size: 12px;
}

.main {
    min-height: calc(100vh - 64px);
    padding: 25px 14px 120px;
    overflow-y: auto;
}

.content {
    width: min(900px, 100%);
    margin: auto;
}

.welcome {
    text-align: center;
    padding: 55px 10px 25px;
}

.welcome h1 {
    font-size: clamp(42px, 10vw, 70px);
    margin: 0;
}

.gradient {
    background: linear-gradient(
        90deg,
        #8876ff,
        #d35cff,
        #62caff
    );
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.welcome p {
    color: #9299a7;
    line-height: 1.6;
}

.cards {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
    max-width: 650px;
    margin: 30px auto;
}

.card {
    border: 1px solid #292d38;
    border-radius: 14px;
    padding: 16px;
    background: #11141b;
    color: #d4d8e0;
    cursor: pointer;
    text-align: left;
}

.card:hover {
    background: #181c25;
}

.message {
    display: flex;
    margin: 18px 0;
}

.user {
    justify-content: flex-end;
}

.bubble {
    max-width: 94%;
    padding: 13px 15px;
    border-radius: 17px;
    line-height: 1.65;
    overflow-wrap: anywhere;
}

.user .bubble {
    background: #202532;
}

.assistant .bubble {
    background: transparent;
}

.codebox {
    margin: 15px 0;
    border: 1px solid #2b303b;
    border-radius: 14px;
    overflow: hidden;
    background: #0b0d12;
}

.codebar {
    display: flex;
    justify-content: space-between;
    padding: 9px 11px;
    background: #12151c;
    color: #9299a7;
    font-size: 12px;
}

.copy {
    border: 0;
    border-radius: 8px;
    padding: 7px 11px;
    background: #272c38;
    color: white;
    cursor: pointer;
}

pre {
    margin: 0;
    padding: 15px;
    overflow-x: auto;
}

code {
    font-family: Consolas, Monaco, monospace;
}

.composer-wrap {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 12px;
    background: linear-gradient(
        to top,
        #08090d 70%,
        transparent
    );
}

.composer {
    width: min(900px, 100%);
    margin: auto;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    padding: 7px;
    border: 1px solid #2a2e39;
    border-radius: 18px;
    background: #12151c;
}

textarea {
    flex: 1;
    min-height: 45px;
    max-height: 180px;
    resize: none;
    border: 0;
    outline: 0;
    background: transparent;
    color: white;
    padding: 11px;
    font-size: 15px;
}

.send {
    width: 46px;
    height: 46px;
    border: 0;
    border-radius: 13px;
    background: linear-gradient(
        135deg,
        #705cff,
        #b94dff
    );
    color: white;
    cursor: pointer;
    font-size: 18px;
}

.send:disabled {
    opacity: .45;
}

@media (max-width: 600px) {

    .cards {
        grid-template-columns: 1fr;
    }

    .welcome {
        padding-top: 35px;
    }

    .bubble {
        max-width: 98%;
    }
}

</style>

</head>

<body>

<header class="header">

<div class="brand">

<div class="logo">M</div>

Matia AI

</div>

<div class="status" id="status">
Ready
</div>

</header>

<main class="main" id="main">

<div class="content" id="content">

<section class="welcome" id="welcome">

<h1>
<span class="gradient">
Matia AI
</span>
</h1>

<p>
Coding • Math • Study • Debugging • Web • AI • Roblox • 50+ experts
</p>

<div class="cards">

<button class="card"
onclick="quickAsk(
'Build me a complete HTML calculator with CSS and JavaScript. Give me the full copy-paste code.'
)">
💻 Build Code
</button>

<button class="card"
onclick="quickAsk(
'Solve a difficult math problem step by step and explain it clearly.'
)">
🧮 Math Expert
</button>

<button class="card"
onclick="quickAsk(
'Teach me Python from beginner to advanced with examples.'
)">
📚 Study Expert
</button>

<button class="card"
onclick="quickAsk(
'Help me debug my code and explain exactly what is wrong.'
)">
🔧 Debug Expert
</button>

</div>

</section>

</div>

</main>

<div class="composer-wrap">

<form class="composer" id="form">

<textarea
id="input"
rows="1"
placeholder="Ask Matia anything..."
></textarea>

<button
class="send"
id="send"
type="submit">
➤
</button>

</form>

</div>

<script>

const input =
    document.getElementById("input");

const send =
    document.getElementById("send");

const content =
    document.getElementById("content");

const main =
    document.getElementById("main");

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
        (crypto.randomUUID)
        ? crypto.randomUUID()
        : String(Date.now());

    localStorage.setItem(
        "matia_conversation_id",
        conversationId
    );
}


function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
}


function renderAnswer(text) {

    const parts =
        text.split(/```([\\s\\S]*?)```/g);

    let html = "";

    for (
        let i = 0;
        i < parts.length;
        i++
    ) {

        if (i % 2 === 1) {

            let code =
                parts[i].trim();

            let language =
                "code";

            const lines =
                code.split("\\n");

            if (
                lines.length > 1 &&
                /^[a-zA-Z0-9+#._-]+$/.test(
                    lines[0].trim()
                )
            ) {

                language =
                    lines[0].trim();

                code =
                    lines
                        .slice(1)
                        .join("\\n")
                        .trim();
            }

            html +=
                '<div class="codebox">' +
                '<div class="codebar">' +
                '<span>' +
                escapeHTML(language) +
                '</span>' +
                '<button class="copy" ' +
                'onclick="copyCode(this)">' +
                '📋 Copy' +
                '</button>' +
                '</div>' +
                '<pre><code>' +
                escapeHTML(code) +
                '</code></pre>' +
                '</div>';

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


function addMessage(role, text) {

    welcome.style.display = "none";

    const message =
        document.createElement("div");

    message.className =
        "message " + role;

    const bubble =
        document.createElement("div");

    bubble.className = "bubble";

    if (role === "assistant") {

        bubble.innerHTML =
            renderAnswer(text);

    } else {

        bubble.innerHTML =
            escapeHTML(text);
    }

    message.appendChild(bubble);

    content.appendChild(message);

    main.scrollTop =
        main.scrollHeight;
}


function showThinking() {

    const message =
        document.createElement("div");

    message.id = "thinking";

    message.className =
        "message assistant";

    message.innerHTML =
        '<div class="bubble">' +
        '🧠 Matia is thinking...' +
        '</div>';

    content.appendChild(message);

    main.scrollTop =
        main.scrollHeight;
}


function hideThinking() {

    const thinking =
        document.getElementById(
            "thinking"
        );

    if (thinking) {
        thinking.remove();
    }
}


async function sendMessage(text) {

    text = text.trim();

    if (!text || send.disabled) {
        return;
    }

    addMessage("user", text);

    input.value = "";

    send.disabled = true;

    status.textContent =
        "Thinking...";

    showThinking();

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

        hideThinking();

        if (!response.ok) {

            addMessage(
                "assistant",
                "❌ " +
                (
                    data.error ||
                    "Server error."
                )
            );

        } else {

            addMessage(
                "assistant",
                data.answer ||
                "No answer received."
            );
        }

    } catch (error) {

        hideThinking();

        addMessage(
            "assistant",
            "❌ Connection error. Check Render logs."
        );

    } finally {

        send.disabled = false;

        status.textContent =
            "Ready";

        input.focus();
    }
}


function quickAsk(text) {
    sendMessage(text);
}


async function copyCode(button) {

    const box =
        button.closest(".codebox");

    const code =
        box.querySelector("code")
            .innerText;

    try {

        await navigator.clipboard
            .writeText(code);

        button.innerText =
            "✓ Copied";

        setTimeout(() => {

            button.innerText =
                "📋 Copy";

        }, 1200);

    } catch (error) {

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

            sendMessage(
                input.value
            );
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

            sendMessage(
                input.value
            );
        }
    }
);


input.addEventListener(
    "input",
    function() {

        this.style.height =
            "auto";

        this.style.height =
            Math.min(
                this.scrollHeight,
                180
            ) + "px";
    }
);

</script>

</body>

</html>
"""


# =========================
# ROUTES
# =========================

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/health")
def health():

    return jsonify({
        "status": "online",
        "name": "Matia AI",
        "model": MODEL,
        "api_configured": bool(GROQ_API_KEY),
        "expert_count": len(EXPERTS)
    })


@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        message = str(
            data.get("message", "")
        ).strip()

        conversation_id = str(
            data.get(
                "conversation_id",
                uuid.uuid4()
            )
        )

        if not message:

            return jsonify({
                "error":
                    "Write a message first."
            }), 400

        if len(message) > MAX_MESSAGE_CHARS:

            return jsonify({
                "error":
                    "Message is too long."
            }), 400

        answer, experts, language = ask_ai(
            conversation_id,
            message
        )

        add_history(
            conversation_id,
            "user",
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
            "conversation_id":
                conversation_id
        })

    except Exception as error:

        print(
            "MATIA ERROR:",
            repr(error)
        )

        return jsonify({
            "error": str(error)
        }), 500


# =========================
# START
# =========================

if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
    

import os
import uuid
from flask import Flask, request, jsonify, render_template_string
from groq import Groq

app = Flask(__name__)

# ============================================================
# MATIA AI CONFIG
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

MODEL = os.getenv(
    "MATIA_MODEL",
    "openai/gpt-oss-120b"
)

MAX_HISTORY = 8
MAX_OUTPUT_TOKENS = 6000
MAX_MESSAGE_LENGTH = 12000

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

conversations = {}


# ============================================================
# 50+ EXPERT AREAS
# ============================================================

EXPERTS = [
    "General AI",
    "Coding Expert",
    "Python Expert",
    "JavaScript Expert",
    "TypeScript Expert",
    "HTML Expert",
    "CSS Expert",
    "Java Expert",
    "C Expert",
    "C++ Expert",
    "C# Expert",
    "PHP Expert",
    "Rust Expert",
    "Go Expert",
    "SQL Expert",
    "Lua Expert",
    "Luau Expert",
    "Roblox Expert",
    "Game Development",
    "Web Development",
    "App Development",
    "AI Expert",
    "Machine Learning",
    "API Expert",
    "Database Expert",
    "Git Expert",
    "GitHub Expert",
    "DevOps Expert",
    "Debugging Expert",
    "Testing Expert",
    "Math Expert",
    "Statistics Expert",
    "Science Expert",
    "Study Expert",
    "Teacher Mode",
    "Quiz Expert",
    "Flashcard Expert",
    "Writing Expert",
    "Editing Expert",
    "Translation Expert",
    "Research Assistant",
    "Summarization",
    "Data Analysis",
    "Logic Expert",
    "Problem Solving",
    "Planning Expert",
    "Brainstorming",
    "Productivity",
    "Security Education",
    "Documentation Expert",
    "Project Architecture"
]


# ============================================================
# EXPERT ROUTING
# ============================================================

ROUTES = {

    "Coding Expert": [
        "code",
        "coding",
        "program",
        "programming",
        "script",
        "function",
        "class",
        "algorithm",
        "developer"
    ],

    "Math Expert": [
        "math",
        "mathematics",
        "calculate",
        "equation",
        "algebra",
        "geometry",
        "fraction",
        "percentage",
        "probability",
        "calculus",
        "derivative",
        "integral"
    ],

    "Study Expert": [
        "study",
        "school",
        "homework",
        "lesson",
        "learn",
        "teach",
        "exam",
        "test",
        "revision"
    ],

    "Debugging Expert": [
        "error",
        "bug",
        "debug",
        "broken",
        "fix",
        "not working",
        "traceback",
        "exception",
        "syntaxerror"
    ],

    "Roblox Expert": [
        "roblox",
        "roblox studio",
        "luau",
        "localscript",
        "serverscript",
        "remoteevent",
        "leaderstats"
    ],

    "Web Development": [
        "website",
        "web app",
        "frontend",
        "backend",
        "html",
        "css",
        "javascript",
        "webpage"
    ],

    "AI Expert": [
        "ai",
        "artificial intelligence",
        "llm",
        "model",
        "prompt",
        "machine learning"
    ],

    "API Expert": [
        "api",
        "endpoint",
        "rest api",
        "webhook",
        "token",
        "json api"
    ],

    "GitHub Expert": [
        "github",
        "git",
        "repository",
        "repo",
        "commit",
        "branch",
        "pull request"
    ],

    "DevOps Expert": [
        "render",
        "deploy",
        "deployment",
        "server",
        "hosting",
        "docker",
        "environment variable"
    ],

    "Writing Expert": [
        "write",
        "writing",
        "essay",
        "email",
        "message",
        "story",
        "caption",
        "paragraph"
    ],

    "Translation Expert": [
        "translate",
        "translation"
    ],

    "Data Analysis": [
        "data",
        "dataset",
        "csv",
        "dataframe",
        "analysis",
        "analyze",
        "chart"
    ],

    "Quiz Expert": [
        "quiz",
        "quiz me",
        "test me",
        "questions"
    ],

    "Flashcard Expert": [
        "flashcard",
        "flashcards",
        "flash card"
    ],

    "Planning Expert": [
        "plan",
        "roadmap",
        "steps",
        "schedule",
        "organize"
    ],

    "Brainstorming": [
        "ideas",
        "idea",
        "brainstorm",
        "suggest"
    ]
}


def detect_experts(text):

    lowered = text.lower()

    detected = []

    for expert, keywords in ROUTES.items():

        for keyword in keywords:

            if keyword in lowered:

                detected.append(expert)

                break

    if not detected:

        detected.append("General AI")

    return detected[:8]


def detect_language(text):

    lowered = text.lower()

    languages = {

        "Python": [
            "python",
            ".py"
        ],

        "JavaScript": [
            "javascript",
            ".js"
        ],

        "TypeScript": [
            "typescript",
            ".ts"
        ],

        "HTML": [
            "html",
            ".html"
        ],

        "CSS": [
            "css",
            ".css"
        ],

        "Java": [
            "java",
            ".java"
        ],

        "C++": [
            "c++",
            "cpp",
            ".cpp"
        ],

        "C#": [
            "c#",
            "csharp",
            ".cs"
        ],

        "PHP": [
            "php",
            ".php"
        ],

        "Rust": [
            "rust",
            ".rs"
        ],

        "Go": [
            "golang",
            ".go"
        ],

        "SQL": [
            "sql",
            ".sql"
        ],

        "Lua/Luau": [
            "lua",
            "luau",
            ".lua"
        ],

        "Bash": [
            "bash",
            ".sh"
        ],

        "PowerShell": [
            "powershell",
            ".ps1"
        ]
    }

    for language, keywords in languages.items():

        for keyword in keywords:

            if keyword in lowered:

                return language

    return "Auto"


def is_code_request(text):

    lowered = text.lower()

    phrases = [

        "write code",
        "give me code",
        "give me the code",
        "make code",
        "create code",
        "generate code",
        "full code",
        "complete code",
        "write a script",
        "make a script",
        "create a script",
        "build a website",
        "make a website",
        "make an app",
        "create an app"
    ]

    if "```" in text:

        return True

    if any(
        phrase in lowered
        for phrase in phrases
    ):

        return True

    return any(
        word in lowered
        for word in [
            "code",
            "script",
            "program",
            "function",
            "html",
            "python",
            "javascript",
            "luau",
            "sql"
        ]
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are MATIA AI.

You are a powerful multi-purpose AI assistant with
50+ expert areas.

Your job is to understand the user's actual goal
and complete it.

GENERAL RULES:

- Answer clearly.
- Be helpful.
- Use conversation context.
- Do not ask unnecessary questions.
- If the request is clear, do it.
- Answer in the user's language when appropriate.
- Never pretend you executed something if you did not.
- Never invent tool results.

CODING EXPERT:

When the user asks for code, ACTUALLY GIVE THE CODE.

Do not only explain how to write it.

If the user asks for an HTML calculator,
give a complete HTML document containing HTML,
CSS and JavaScript.

If the user asks for Python,
give actual Python code.

If the user says "full code",
give the complete implementation.

Code must be:

- complete
- copy-paste ready
- useful
- readable
- syntactically valid
- reasonably robust

Always put code inside Markdown code blocks.

DEBUGGING:

When the user gives an error:

1. Find the likely cause.
2. Explain it briefly.
3. Give the exact fix.
4. Provide corrected code when appropriate.

MATH:

Solve carefully and show useful steps.

STUDY:

Teach clearly using examples and explanations.

SMART CONTEXT:

Understand references such as:

"that code"
"same one"
"make it better"
"continue"
"add another button"

EXPERT AREAS INCLUDE:

coding,
Python,
JavaScript,
TypeScript,
HTML,
CSS,
Java,
C,
C++,
C#,
PHP,
Rust,
Go,
SQL,
Lua,
Luau,
Roblox,
web development,
app development,
AI,
APIs,
databases,
GitHub,
DevOps,
debugging,
testing,
math,
statistics,
science,
studying,
writing,
translation,
research,
data analysis,
planning,
productivity
and more.

For cybersecurity requests,
stay within legal, authorized,
defensive and educational use.
"""


# ============================================================
# MEMORY
# ============================================================

def get_history(conversation_id):

    if conversation_id not in conversations:

        conversations[conversation_id] = []

    return conversations[conversation_id]


def add_history(
    conversation_id,
    role,
    content
):

    history = get_history(
        conversation_id
    )

    history.append({

        "role": role,

        "content": content

    })

    if len(history) > MAX_HISTORY:

        conversations[conversation_id] = \
            history[-MAX_HISTORY:]


def build_messages(
    conversation_id,
    user_message
):

    experts = detect_experts(
        user_message
    )

    language = detect_language(
        user_message
    )

    code_mode = is_code_request(
        user_message
    )

    routing = f"""

CURRENT ROUTING:

Experts:
{", ".join(experts)}

Programming language:
{language}

Code mode:
{"YES" if code_mode else "NO"}
"""

    messages = [

        {
            "role": "system",
            "content":
                SYSTEM_PROMPT +
                routing
        }

    ]

    for item in get_history(
        conversation_id
    ):

        messages.append({

            "role": item["role"],

            "content": item["content"]

        })

    messages.append({

        "role": "user",

        "content": user_message

    })

    return (
        messages,
        experts,
        language,
        code_mode
    )


# ============================================================
# GROQ
# ============================================================

def ask_ai(
    conversation_id,
    user_message
):

    if client is None:

        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add GROQ_API_KEY to Render "
            "Environment Variables."
        )

    (
        messages,
        experts,
        language,
        code_mode
    ) = build_messages(
        conversation_id,
        user_message
    )

    response = client.chat.completions.create(

        model=MODEL,

        messages=messages,

        max_tokens=MAX_OUTPUT_TOKENS,

        temperature=0.35
    )

    answer = (
        response
        .choices[0]
        .message
        .content
    )

    if not answer:

        answer = (
            "I couldn't generate an answer."
        )

    return (
        answer,
        experts,
        language,
        code_mode
    )


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
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

    color: #f5f7fb;

    font-family:
        Arial,
        Helvetica,
        sans-serif;
}

.header {

    height: 64px;

    padding: 0 18px;

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    background: #0d0f15;

    border-bottom:
        1px solid #252936;

    position: sticky;

    top: 0;

    z-index: 20;
}

.brand {

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 19px;

    font-weight: 700;
}

.logo {

    width: 38px;

    height: 38px;

    border-radius: 12px;

    display: flex;

    align-items: center;

    justify-content: center;

    background:
        linear-gradient(
            135deg,
            #705cff,
            #c64dff
        );

    font-weight: bold;
}

.status {

    color: #8b92a0;

    font-size: 12px;
}

.main {

    min-height:
        calc(100vh - 64px);

    padding:
        25px 14px 125px;

    overflow-y: auto;
}

.content {

    width:
        min(900px, 100%);

    margin: auto;
}

.welcome {

    text-align: center;

    padding:
        55px 10px 25px;
}

.welcome h1 {

    font-size:
        clamp(40px, 9vw, 68px);

    margin: 0;
}

.gradient {

    background:
        linear-gradient(
            90deg,
            #8876ff,
            #d35cff,
            #62caff
        );

    -webkit-background-clip:
        text;

    background-clip:
        text;

    color: transparent;
}

.welcome p {

    color: #9299a7;

    line-height: 1.6;
}

.cards {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 10px;

    max-width: 650px;

    margin: 30px auto;
}

.card {

    text-align: left;

    border:
        1px solid #292d38;

    border-radius: 14px;

    padding: 15px;

    background: #11141b;

    color: #d4d8e0;

    cursor: pointer;
}

.card:hover {

    background: #181c25;
}

.message {

    display: flex;

    margin: 18px 0;
}

.user {

    justify-content:
        flex-end;
}

.bubble {

    max-width: 94%;

    padding:
        13px 15px;

    border-radius: 17px;

    line-height: 1.65;

    overflow-wrap:
        anywhere;
}

.user .bubble {

    background:
        #202532;
}

.assistant .bubble {

    background:
        transparent;
}

.codebox {

    margin: 15px 0;

    overflow: hidden;

    border:
        1px solid #2b303b;

    border-radius: 14px;

    background:
        #0b0d12;
}

.codebar {

    display: flex;

    align-items: center;

    justify-content:
        space-between;

    padding:
        9px 11px;

    background:
        #12151c;

    color: #9299a7;

    font-size: 12px;
}

.copy {

    border: 0;

    border-radius: 8px;

    padding:
        7px 11px;

    background:
        #272c38;

    color: white;

    cursor: pointer;
}

pre {

    margin: 0;

    padding: 15px;

    overflow-x: auto;
}

code {

    font-family:
        Consolas,
        Monaco,
        monospace;
}

.composer-wrap {

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

    width:
        min(900px, 100%);

    margin: auto;

    display: flex;

    gap: 8px;

    align-items:
        flex-end;

    padding: 7px;

    border:
        1px solid #2a2e39;

    border-radius: 18px;

    background:
        #12151c;
}

textarea {

    flex: 1;

    min-height: 45px;

    max-height: 180px;

    resize: none;

    border: 0;

    outline: 0;

    background:
        transparent;

    color: white;

    padding: 11px;

    font-size: 15px;
}

.send {

    width: 46px;

    height: 46px;

    border: 0;

    border-radius: 13px;

    background:
        linear-gradient(
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

        grid-template-columns:
            1fr;
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

<div class="logo">
M
</div>

Matia AI

</div>

<div class="status"
id="status">

Ready

</div>

</header>

<main
class="main"
id="main">

<div
class="content"
id="content">

<section
class="welcome"
id="welcome">

<h1>

<span class="gradient">

Matia AI

</span>

</h1>

<p>

Coding • Math • Study •
Debugging • Web • AI •
Roblox • 50+ expert areas

</p>

<div class="cards">

<button
class="card"
onclick="quickAsk(
'Build me a complete HTML calculator with CSS and JavaScript. Give me the full copy-paste code.'
)">

💻 Build Code

</button>

<button
class="card"
onclick="quickAsk(
'Solve a difficult math problem step by step and explain it clearly.'
)">

🧮 Math Expert

</button>

<button
class="card"
onclick="quickAsk(
'Teach me Python from beginner to advanced with examples.'
)">

📚 Study Expert

</button>

<button
class="card"
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

<form
class="composer"
id="form">

<textarea
id="input"
rows="1"
placeholder="Ask Matia anything..."></textarea>

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
    document.getElementById(
        "input"
    );

const send =
    document.getElementById(
        "send"
    );

const content =
    document.getElementById(
        "content"
    );

const main =
    document.getElementById(
        "main"
    );

const welcome =
    document.getElementById(
        "welcome"
    );

const status =
    document.getElementById(
        "status"
    );

let conversationId =
    localStorage.getItem(
        "matia_conversation_id"
    );

if (!conversationId) {

    conversationId =
        crypto.randomUUID
        ? crypto.randomUUID()
        : String(Date.now());

    localStorage.setItem(
        "matia_conversation_id",
        conversationId
    );
}


function escapeHTML(value) {

    return String(value)
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
}


function renderAnswer(text) {

    const parts =
        text.split(
            /```([\s\S]*?)```/g
        );

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
                code.split("\n");

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
                        .join("\n")
                        .trim();
            }

            html += `

<div class="codebox">

<div class="codebar">

<span>
${escapeHTML(language)}
</span>

<button
class="copy"
onclick="copyCode(this)">

📋 Copy

</button>

</div>

<pre><code>${escapeHTML(
    code
)}</code></pre>

</div>

`;

        } else {

            let normal =
                escapeHTML(parts[i]);

            normal =
                normal.replace(
                    /\*\*(.*?)\*\*/g,
                    "<strong>$1</strong>"
                );

            normal =
                normal.replace(
                    /`([^`]+)`/g,
                    "<code>$1</code>"
                );

            normal =
                normal.replace(
                    /\n/g,
                    "<br>"
                );

            html += normal;
        }
    }

    return html;
}


function addMessage(
    role,
    text
) {

    welcome.style.display =
        "none";

    const message =
        document.createElement(
            "div"
        );

    message.className =
        "message " + role;

    const bubble =
        document.createElement(
            "div"
        );

    bubble.className =
        "bubble";

    if (
        role === "assistant"
    ) {

        bubble.innerHTML =
            renderAnswer(text);

    } else {

        bubble.innerHTML =
            escapeHTML(text);
    }

    message.appendChild(
        bubble
    );

    content.appendChild(
        message
    );

    main.scrollTop =
        main.scrollHeight;
}


function showThinking() {

    const message =
        document.createElement(
            "div"
        );

    message.id =
        "thinking";

    message.className =
        "message assistant";

    message.innerHTML = `

<div class="bubble">

🧠 Matia is thinking...

</div>

`;

    content.appendChild(
        message
    );

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


async function sendMessage(
    text
) {

    text = text.trim();

    if (
        !text ||
        send.disabled
    ) {

        return;
    }

    addMessage(
        "user",
        text
    );

    input.value = "";

    send.disabled = true;

    status.textContent =
        "Thinking...";

    showThinking();

    try {

        const response =
            await fetch(
                "/chat",
                {

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

                }
            );

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

        send.disabled =
            false;

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
        button.closest(
            ".codebox"
        );

    const code =
        box.querySelector(
            "code"
        ).innerText;

    try {

        await navigator.clipboard
            .writeText(code);

        button.innerText =
            "✓ Copied";

        setTimeout(
            function() {

                button.innerText =
                    "📋 Copy";

            },
            1200
        );

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


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():

    return render_template_string(
        HTML
    )


@app.route("/health")
def health():

    return jsonify({

        "status": "online",

        "name": "Matia AI",

        "model": MODEL,

        "api_configured":
            bool(GROQ_API_KEY),

        "expert_count":
            len(EXPERTS)

    })


@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    try:

        data =
            request.get_json(
                silent=True
            ) or {}

        message = str(
            data.get(
                "message",
                ""
            )
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

        if len(message) > MAX_MESSAGE_LENGTH:

            return jsonify({

                "error":
                    "Message is too long."

            }), 400

        (
            answer,
            experts,
            language,
            code_mode
        ) = ask_ai(
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

            "code_mode": code_mode,

            "conversation_id":
                conversation_id

        })

    except Exception as error:

        print(
            "MATIA ERROR:",
            repr(error)
        )

        return jsonify({

            "error":
                str(error)

        }), 500


if __name__ == "__main__":

    port = int(
        os.getenv(
            "PORT",
            "5000"
        )
    )

    print("=" * 60)

    print("MATIA AI")

    print("=" * 60)

    print(
        "Model:",
        MODEL
    )

    print(
        "Experts:",
        len(EXPERTS)
    )

    print(
        "Groq:",
        "CONNECTED"
        if client
        else "MISSING API KEY"
    )

    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )

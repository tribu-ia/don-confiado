import readline from "readline";

type ChatResponse = {
    reply?: string;
    [key: string]: unknown;
};

async function callBackend(message: string, userId: string): Promise<ChatResponse> {
    const endpoint = process.env.BACKEND_URL || "http://127.0.0.1:8000/api/chat_v2.0";
    const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, user_id: userId })
    });
    if (!response.ok) {
        throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }
    return response.json();
}

async function main() {
    const userId = process.env.USER_ID || "cli-user";
    console.log("Simple Chat CLI (type 'exit' to quit)");
    console.log(`Using backend: ${process.env.BACKEND_URL || "http://127.0.0.1:8000/api/chat_v2.0"}`);

    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        terminal: true
    });

    const ask = (q: string) => new Promise<string>((resolve) => rl.question(q, resolve));

    try {
        while (true) {
            const text = (await ask("You: ")).trim();
            if (!text) continue;
            if (text.toLowerCase() === "exit") break;

            try {
                const result = await callBackend(text, userId);
                const reply = result.reply ?? "(no reply)";
                console.log(`Bot: ${reply}`);
            } catch (err: any) {
                console.error(`Error: ${err?.message || String(err)}`);
            }
        }
    } finally {
        rl.close();
    }
}

main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
});



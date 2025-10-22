import readline from "readline";

type ChatResponse = {
    answer?: string;
    reply?: string;
    [key: string]: unknown;
};

async function callBackend(message: string, userId: string): Promise<ChatResponse> {
    const endpoint = process.env.BACKEND_URL || "http://127.0.0.1:8000/api/graphrag/enhanced/ask";
    const params = new URLSearchParams({
        query: message,
        retrieval_method: "hybrid",
        top_k: "5",
        use_graphrag: "true"
    });
    
    const response = await fetch(`${endpoint}?${params}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" }
    });
    if (!response.ok) {
        throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }
    return response.json() as Promise<ChatResponse>;
}

async function main() {
    const userId = process.env.USER_ID || "cli-user";
    console.log("Simple Chat CLI (type 'exit' to quit)");
    console.log(`Using backend: ${process.env.BACKEND_URL || "http://127.0.0.1:8000/api/graphrag/enhanced/ask"}`);

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
                const reply = result.answer ?? "(no reply)";
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



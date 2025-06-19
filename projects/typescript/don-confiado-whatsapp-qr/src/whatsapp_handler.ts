import { WASocket, DisconnectReason } from "baileys";

class WhatsAppHandler {
    private sock!: WASocket;
    private qrAttempts = 0;
    private readonly maxQrAttempts = 3;
    private readonly saveCreds: () => Promise<void>;

    constructor(sock: WASocket, saveCreds: () => Promise<void>) {
        this.sock = sock;
        this.saveCreds = saveCreds;

        // Bind methods to this instance
        this.onCredsUpdate = this.onCredsUpdate.bind(this);
        this.onMessagesUpsert = this.onMessagesUpsert.bind(this);
        this.onConnectionUpdate = this.onConnectionUpdate.bind(this);

        sock.ev.on("creds.update", this.onCredsUpdate.bind(this));
        sock.ev.on("messages.upsert", this.onMessagesUpsert.bind(this));
        sock.ev.on("connection.update", this.onConnectionUpdate.bind(this));
    }

    onCredsUpdate() {
        this.saveCreds();
    }

    onMessagesUpsert(m: any) {
        console.log("--------------------[ sock.ev.on - messages.upsert ]-------------------------");
        console.log("message.upsert:", m);
        for (const msg of m.messages) {
            if (!msg.key.fromMe) {
                console.log("Mensaje recibido de:", msg.key.remoteJid);
                console.log("Contenido del mensaje:", msg.message);
            }
        }
        console.log("Messages:", m.messages);
        console.log("-----------------------------------------------------------");
    }

    async onConnectionUpdate(update: any) {
        console.log("--------------------[ sock.ev.on - connection.update ]-------------------------");
        console.log("Connection update:", update);
        console.log("-----------------------------------------------------------");

        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            this.qrAttempts++;
            if (this.qrAttempts > this.maxQrAttempts) {
                console.log("❌ Demasiados intentos de escaneo de QR. Cerrando conexión...");
                await this.sock.logout();
                process.exit(1);
                return;
            }

            QRCode.toString(qr, { type: "terminal", small: true }, (err, url) => {
                if (err) return console.error("Error generating QR:", err);
                console.log(url);
                console.log(`📱 Escanea el código QR (${this.qrAttempts}/${this.maxQrAttempts})`);
            });
        }

        if (connection === "open") {
            console.log("✅ Conectado a WhatsApp");
        } else if (connection === "close") {
            console.log("❌ Conexión cerrada");
            const shouldReconnect = (lastDisconnect?.error as any)?.output?.statusCode !== DisconnectReason.loggedOut;
            console.log("⚠️ Desconectado de WhatsApp", lastDisconnect?.error);
            if (shouldReconnect) {
                console.log("🔁 Reintentando conexión...");
                try {
                    await main(); // Reconectar
                } catch (err) {
                    console.error("❌ Error reconectando:", err);
                    process.exit(1);
                }
            } else {
                console.log("🚪 Sesión cerrada");
                process.exit(0);
            }
        }
    }
}


export { WhatsAppHandler };
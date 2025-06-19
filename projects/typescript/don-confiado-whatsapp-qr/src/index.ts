import P from "pino";
import * as QRCode from "qrcode";
import {
    makeWASocket,
    useMultiFileAuthState,
    DisconnectReason,
    SocketConfig,
    WASocket,
    AuthenticationState
} from "baileys";

// Tipado correcto del recurso global
interface GlobalResource {
    authState: AuthenticationState | null;
    authSaveCreds: (() => Promise<void>) | null;
}

const globalResource: GlobalResource = {
    authState: null,
    authSaveCreds: null
};

async function main() {
    const sock = makeWASocket({ auth: globalResource.authState as AuthenticationState });
    const handler = new WhatsAppHandler(sock, globalResource.authSaveCreds as () => Promise<void>);

    sock.ev.on("creds.update", handler.onCredsUpdate);
    sock.ev.on("messages.upsert", handler.onMessagesUpsert);
    sock.ev.on("connection.update", handler.onConnectionUpdate);
}

export async function startAuthState(): Promise<void> {
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    globalResource.authState = state;
    globalResource.authSaveCreds = saveCreds;
    main().catch((err) => {
        console.error("Error en startAuthState:", err);
        process.exit(1);
    })
}

async function startWhatsApp() {
    try {
        await startAuthState();
        console.log("✅ Estado de autenticación inicializado correctamente.");
    } catch (error) {
        console.error("❌ Error al iniciar el estado de autenticación:", error);
        process.exit(1);
    }
}

export { globalResource };
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




startWhatsApp().catch((err) => {
    console.error("❌ Error al iniciar WhatsApp:", err);
    process.exit(1);
});
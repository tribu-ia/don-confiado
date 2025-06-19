import P from "pino";
import { makeWASocket, useMultiFileAuthState, DisconnectReason, SocketConfig } from "baileys";
import * as QRCode from "qrcode";

async function main() {
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    const sock = makeWASocket({ auth: state });

    
    let qrAttempts = 0;
    const maxQrAttempts = 3;


    sock.ev.on("creds.update", saveCreds);

    sock.ev.on("messages.upsert", (m) => {

        console.log("--------------------[ sock.ev.on - messages.upsert ]-------------------------");
        console.log("message.upsert:", m);
        console.log("-----------------------------------------------------------");

    });


    sock.ev.on("connection.update", async (update) => {
        console.log("--------------------[ sock.ev.on - connection.update ]-------------------------");
        console.log("Connection update:", update);
        console.log("-----------------------------------------------------------");
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            qrAttempts++;
            if (qrAttempts > maxQrAttempts) {
                console.log("❌ Demasiados intentos de escaneo de QR. Cerrando conexión...");
                await sock.logout();  // Esto elimina las credenciales
                process.exit(1); // Termina el proceso
                return;
            }

            QRCode.toString(qr, { type: "terminal", small: true }, (err, url) => {
                if (err) return console.error("Error generating QR:", err);
                console.log(url);
                console.log(`📱 Escanea el código QR (${qrAttempts}/${maxQrAttempts})`);
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
                    main();
                } catch (err) {
                    console.error("❌ Error reconectando:", err);
                    process.exit(1);
                }
            } else {
                console.log("🚪 Sesión cerrada");
                process.exit(0);
            }
        }
    });
}

main().catch((err) => {
    console.error("Error en main:", err);
    process.exit(1);
});

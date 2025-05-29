import P from "pino";
import { makeWASocket, useMultiFileAuthState, DisconnectReason, SocketConfig } from "baileys";
import * as QRCode from "qrcode";
async function main (){
    // DO NOT USE IN PROD!!!!
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    // will use the given state to connect
    // so if valid credentials are available -- it'll connect without QR
    const sock = makeWASocket({ auth: state });
    // this will be called as soon as the credentials are updated
    sock.ev.on("creds.update", saveCreds);
    // Guardar las credenciales en disco
    sock.ev.on("creds.update", saveCreds);

    // Manejo de conexión
    sock.ev.on("connection.update", (update) => {
        const { connection, lastDisconnect, qr } = update;

        if (qr) {
            // Generar QR en consola
            QRCode.toString(qr, { type: "terminal" }, (err, url) => {
                if (err) return console.error("Error generating QR:", err);
                console.log(url);
            });
        }

        if (connection === "open") {
            console.log("✅ Connected to WhatsApp");
        } else if (connection === "close") {
            const shouldReconnect = (lastDisconnect?.error as any)?.output?.statusCode !== DisconnectReason.loggedOut;
            console.log("⚠️ Disconnected from WhatsApp", lastDisconnect?.error);
            if (shouldReconnect) {
                console.log("🔁 Reconnecting...");
                sock.connect();
            } else {
                console.log("🚪 Logged out");
            }
        }
    });
}
main().catch((err) => {
    console.error("Error in main:", err);
});
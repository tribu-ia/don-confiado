import P from "pino";
import * as QRCode from "qrcode";
import { WhatsAppHandler } from "./whatsapp_handler";
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

;
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




startWhatsApp().catch((err) => {
    console.error("❌ Error al iniciar WhatsApp:", err);
    process.exit(1);
});
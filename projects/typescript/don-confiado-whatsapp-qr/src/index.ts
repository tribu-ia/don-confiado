import P from "pino";
import * as QRCode from "qrcode";
import { globalResource } from "./global_resource"; 
import { WhatsAppHandler } from "./whatsapp_handler";

import {
    makeWASocket,
    useMultiFileAuthState,
    DisconnectReason,
    SocketConfig,
    WASocket,
    AuthenticationState
} from "baileys";
import { rmSync, existsSync } from "fs";
import { join } from "path";

async function main() {
    const sock = makeWASocket({ auth: globalResource.authState as AuthenticationState });
    const handler = new WhatsAppHandler(sock, globalResource.authSaveCreds as () => Promise<void>,main);

}



function deleteAuthFolder(folderName: string) {
    const fullPath = join(process.cwd(), folderName);
    console.log(`🗑️ Eliminando carpeta de autenticación: ${fullPath}`);
    if (existsSync(fullPath)) {
        rmSync(fullPath, { recursive: true, force: true });
        console.log(`🗑️ Carpeta "${folderName}" eliminada correctamente.`);
    } else {
        console.log(`⚠️ La carpeta "${folderName}" no existe.`);
    }
}


export async function startAuthState(): Promise<void> {
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    console.log("Estado de autenticación cargado correctamente. ",state);

    if (!state.registered && state.creds.me) {

        console.log("🔄 La sesión no está registrada. Eliminando archivos de autenticación...");
        deleteAuthFolder("auth_info_baileys");
        console.log("🔄 Reiniciando el estado de autenticación...");
        const { state: newState, saveCreds: newSaveCreds } = await useMultiFileAuthState("auth_info_baileys");
        console.log("🔄 Nuevo estado de autenticación creado.");
    }
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
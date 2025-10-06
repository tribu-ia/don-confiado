import { WhatsAppHandler } from "./whatsapp_handler";

async function main() {
    const mode = process.argv[2]?.toLowerCase();
    if (mode === "chat") {
        await import("./chat_cli");
        return; // chat_cli starts its own main
    }

    const handler = new WhatsAppHandler();
    handler.initSocket();
}


/*

export async function startAuthState(): Promise<void> {
    const { state, saveCreds } = await useMultiFileAuthState("auth_info_baileys");
    console.log("🔄 Cargando el estado de autenticación desde 'auth_info_baileys'");
    console.log("Estado de autenticación cargado correctamente. state:\n",state);
    console.log("----------------------------------------------------------------");

    if (!state.registered && state.creds.me) {

        console.log("🔄 La sesión no está registrada. Eliminando archivos de autenticación...");
        deleteAuthFolder("auth_info_baileys");
        console.log("🔄 Reiniciando el estado de autenticación...");
        
        console.log("🔄 Nuevo estado de autenticación creado.");
    }
    globalResource.authState = state;
    globalResource.authSaveCreds = saveCreds;
    main().catch((err) => {
        console.error("Error en startAuthState:", err);
        process.exit(1);
    })
}
*/

main().catch((err) => {
    console.error("❌ Error en la función principal:", err);
    process.exit(1);    
});



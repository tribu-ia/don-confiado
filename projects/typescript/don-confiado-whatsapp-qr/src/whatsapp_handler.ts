
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
import { rmSync, existsSync } from "fs";
import { join } from "path";

class WhatsAppHandler {
    private sock!: WASocket;
    private qrAttempts = 0;
    private readonly maxQrAttempts = 3;
    private  saveCreds: () => Promise<void> | null;
    //private readonly restartSock: () => Promise<void>;
    private authState: AuthenticationState | undefined;

    async initSocket() {
        console.log("🔄 Inicializando socket de WhatsApp...");  
        const { state: newState, saveCreds: newSaveCreds } = await useMultiFileAuthState("auth_info_baileys");
        this.authState = newState;
        this.saveCreds = newSaveCreds;
        this.sock  = makeWASocket({ auth: this.authState as AuthenticationState });;
        this.sock.ev.on("creds.update", this.onCredsUpdate.bind(this));
        this.sock.ev.on("messages.upsert", this.onMessagesUpsert.bind(this));
        this.sock.ev.on("connection.update", this.onConnectionUpdate.bind(this));
    }

    async initSaveCredentials() {
        
        
    }

    constructor() {
        // Bind methods to this instance
        this.saveCreds = null;
        this.onCredsUpdate = this.onCredsUpdate.bind(this);
        this.onMessagesUpsert = this.onMessagesUpsert.bind(this);
        this.onConnectionUpdate = this.onConnectionUpdate.bind(this);        
    }



    onCredsUpdate(q:any) {
        
        console.log("--------------------[ onCredsUpdate  ]-------------------------");
        console.log("Credenciales actualizadas:", q);
        console.log("-----------------------------------------------------------");
        
        this.saveCreds();
    }
    /**
     * Maneja los mensajes entrantes y los muestra en la consola.
     * @param m - El objeto de mensajes recibido.
     */
    onMessagesUpsert(m: any) {
        console.log("--------------------[ sock.ev.on - messages.upsert ]-------------------------");
        //console.log("message.upsert:", m);
        for (const msg of m.messages) {
            console.log("Mensaje recibido:\n", msg);
            if (msg.key.fromMe) {           
                console.log("\tIgnorando mensaje enviado por el propio cliente:", msg.key.remoteJid);     
                continue; // Ignorar mensajes enviados por el propio cliente
            }
            try {
                if (msg.message){
                    console.log("Mensaje recibido de:", msg.key.remoteJid);
                    console.log("Contenido del mensaje:", msg.message.conversation || msg.message.extendedTextMessage?.text || "No texto disponible");
                    //Marcar mensaje como leído
                    this.sock.readMessages([msg.key]);
                    // Echo del mensaje
                    this.sock.sendMessage(msg.key.remoteJid, {
                        text: `Echo: ${msg.message.conversation || msg.message.extendedTextMessage?.text || "No texto disponible"}`
                    });
                }
                
            }catch (error) {
                console.error("Error al obtener el ID del mensaje:", msg);
            }
            console.log("- - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - ");
        }
        //console.log("Messages:", m.messages);
        console.log("-----------------------------------------------------------");
    }
    async onConnectionUpdateQR(qr: string) {
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

    async onConnectionUpdateClose(connection: string,lastDisconnect: { error: any; }){
        console.log("❌ Conexión cerrada",((lastDisconnect?.error as any)?.output?.statusCode));
        const shouldReconnect = (lastDisconnect?.error as any)?.output?.statusCode !== DisconnectReason.loggedOut;
        console.log("⚠️ Desconectado de WhatsApp reconnect",shouldReconnect,"Error:", lastDisconnect?.error);
        if (shouldReconnect) {
            console.log("🔁 Reintentando conexión...");
            try {
                await this.initSocket(); // Reconectar
            } catch (err) {
                console.error("❌ Error reconectando:", err);
                process.exit(1);
            }
        } else {
            console.log("🚪 Sesión cerrada");
            this.deleteAuthFolder("auth_info_baileys");
            process.exit(0);
            //this.restartSock();
        }
    }
    async onConnectionUpdate(update: {connection: string; lastDisconnect?: { error: any; }; qr?: string; }) {
        console.log("--------------------[ sock.ev.on - connection.update ]-------------------------");
        console.log("Connection update:", update);
        console.log("-----------------------------------------------------------");

        const { connection , lastDisconnect, qr } = update;
        

        if (qr) {
            this.onConnectionUpdateQR(qr);
        }

        if (connection  === "open") {
            console.log("✅ Conectado a WhatsApp");
        } else if (connection === "close") {
            this.onConnectionUpdateClose(connection,lastDisconnect)
        }
    }


    deleteAuthFolder(folderName: string) {
        const fullPath = join(process.cwd(), folderName);
        console.log(`🗑️ Eliminando carpeta de autenticación: ${fullPath}`);
        if (existsSync(fullPath)) {
            rmSync(fullPath, { recursive: true, force: true });
            console.log(`🗑️ Carpeta "${folderName}" eliminada correctamente.`);
        } else {
            console.log(`⚠️ La carpeta "${folderName}" no existe.`);
        }
    }
}


export { WhatsAppHandler };
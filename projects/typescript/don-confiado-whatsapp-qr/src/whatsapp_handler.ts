import P from "pino";
import * as QRCode from "qrcode";
import {
  makeWASocket,
  useMultiFileAuthState,
  DisconnectReason,
  SocketConfig,
  WASocket,
  AuthenticationState,
  downloadMediaMessage,
  getContentType
} from "baileys";


//import makeWASocket, { downloadMediaMessage } from "@whiskeysockets/baileys"
import {createWriteStream, readFileSync} from "fs";


import { rmSync, existsSync } from "fs";
import { join } from "path";


function fileToBase64(path: string): string {
  const fileBuffer = readFileSync(path);
  return fileBuffer.toString('base64');
}


class WhatsAppHandler {
  private sock!: WASocket;
  private qrAttempts = 0;
  private readonly maxQrAttempts = 3;
  private saveCreds: () => Promise<void> | null;
  //private readonly restartSock: () => Promise<void>;
  private authState: AuthenticationState | undefined;

  async initSocket() {
    console.log("🔄 Inicializando socket de WhatsApp...");
    const { state: newState, saveCreds: newSaveCreds } =
      await useMultiFileAuthState("auth_info_baileys");
    this.authState = newState;
    this.saveCreds = newSaveCreds;
    this.sock = makeWASocket({ auth: this.authState as AuthenticationState });
    this.sock.ev.on("creds.update", this.onCredsUpdate.bind(this));
    this.sock.ev.on("messages.upsert", this.onMessagesUpsert.bind(this));
    this.sock.ev.on("connection.update", this.onConnectionUpdate.bind(this));
  }

  async initSaveCredentials() {}

  constructor() {
    // Bind methods to this instance
    this.saveCreds = async () => {};
    this.onCredsUpdate = this.onCredsUpdate.bind(this);
    this.onMessagesUpsert = this.onMessagesUpsert.bind(this);
    this.onConnectionUpdate = this.onConnectionUpdate.bind(this);
  }

  onCredsUpdate(q: any) {
    console.log(
      "--------------------[ onCredsUpdate  ]-------------------------"
    );
    console.log("Credenciales actualizadas:", q);
    console.log("-----------------------------------------------------------");

    this.saveCreds();
  }

  // Helper function to download and save media files
   downloadAndSaveMedia = (stream: any, filepath: string): Promise<void> => {
    return new Promise((resolve, reject) => {
      const writeStream = createWriteStream(filepath);
      stream.pipe(writeStream);
      writeStream.on('finish', () => {
        console.log(`✅ File saved successfully: ${filepath}`);
        resolve();
      });
      writeStream.on('error', (err) => {
        console.error(`❌ Error saving file: ${err}`);
        reject(err);
      });
    });
  }
  /**
   * Maneja los mensajes entrantes y los muestra en la consola.
   * @param m - El objeto de mensajes recibido.
   */
  async onMessagesUpsert(message_array: any) {
    console.log("--------------------[ sock.ev.on - messages.upsert ]-------------------------");
    //console.log("message.upsert:", m);
    for (const msg of message_array.messages) {
      console.log("Mensaje recibido:\n", msg);
      if (msg.key.fromMe) {
        console.log("\tIgnorando mensaje enviado por el propio cliente:",msg.key.remoteJid);
        continue; // Ignorar mensajes enviados por el propio cliente
      }
      
      try {
        if (msg.message) {
          console.log("Mensaje recibido de:", msg.key.remoteJid);

          const messageType = getContentType(msg.message);
          console.log("Tipo de mensaje:", messageType);
          let  mime_type = "";
          let filename = "";
          let img_caption = "";
          
          
          
          if (messageType === 'imageMessage') {
            mime_type = msg.message.imageMessage.mimetype;
            filename = "/tmp/downloaded-image." + mime_type.split('/')[1];
             
            // download the media as a stream
            const stream = await this.downloadMediaMessage(
                msg,
                'stream',
                {},
                {
                    logger: P({ level: "silent" }),
                    reuploadRequest: this.sock.updateMediaMessage
                }
            );
    
            // save the image file locally and wait for it to finish
            await this.downloadAndSaveMedia(stream, filename);
          }
          if (messageType === 'audioMessage') {
            mime_type = msg.message.audioMessage.mimetype;
            filename = "/tmp/downloaded-audio." + mime_type.split('/')[1];

            const stream = await downloadMediaMessage(
              msg,
              'stream',
              {},
              {
                  logger: P({ level: "silent" }),
                  reuploadRequest: this.sock.updateMediaMessage
              }
            );
  
            // save the audio file locally and wait for it to finish
            await downloadAndSaveMedia(stream, filename);
          }

          console.log(
            "Contenido del mensaje:",
            msg.message.conversation ||
              msg.message.extendedTextMessage?.text ||
              "No texto disponible"
          );

          this.sock.readMessages([msg.key]);

          const message = msg.message.imageMessage?.caption ||
                          msg.message.conversation ||
                          msg.message.extendedTextMessage?.text ||
                          "No texto disponible";


          fetch("http://127.0.0.1:8000/api/chat_v2.0", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              message: message,
              user_id: msg.key.remoteJid,
              mime_type: mime_type,
              file_base64: mime_type ? fileToBase64(filename) : null
            }),
            redirect: "follow",
          })
            .then((response) => response.json())
            .then((response: any) => {
              console.log("API response:", response);

              this.sock.sendMessage(msg.key.remoteJid, {
                text: response.reply ?? "⚠️ No pude entender tu mensaje - juriel",
              });
            })
            .catch((error) => console.error(error));
        }
      } catch (error) {
        console.log(error);
        console.error("Error al obtener el ID del mensaje:", msg);
      }
      console.log(
        "- - - - - - - - - - -  - - - - - - - - - - - - - - - - - - - - - "
      );
    }
    //console.log("Messages:", m.messages);
    console.log("-----------------------------------------------------------");
  }
  async onConnectionUpdateQR(qr: string) {
    this.qrAttempts++;
    if (this.qrAttempts > this.maxQrAttempts) {
      console.log(
        "❌ Demasiados intentos de escaneo de QR. Cerrando conexión..."
      );
      await this.sock.logout();
      process.exit(1);
      return;
    }

    QRCode.toString(qr, { type: "terminal", small: true }, (err, url) => {
      if (err) return console.error("Error generating QR:", err);
      console.log(url);
      console.log(
        `📱 Escanea el código QR (${this.qrAttempts}/${this.maxQrAttempts})`
      );
    });
  }

  async onConnectionUpdateClose(
    connection: string | undefined,
    lastDisconnect: { error: any } | undefined
  ) {
    console.log(
      "❌ Conexión cerrada",
      (lastDisconnect?.error as any)?.output?.statusCode
    );
    const shouldReconnect =
      (lastDisconnect?.error as any)?.output?.statusCode !==
      DisconnectReason.loggedOut;
    console.log(
      "⚠️ Desconectado de WhatsApp reconnect",
      shouldReconnect,
      "Error:",
      lastDisconnect?.error
    );
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
  async onConnectionUpdate(update: {
    connection?: string;
    lastDisconnect?: { error: any };
    qr?: string;
  }) {
    console.log(
      "--------------------[ sock.ev.on - connection.update ]-------------------------"
    );
    console.log("Connection update:", update);
    console.log("-----------------------------------------------------------");

    const { connection, lastDisconnect, qr } = update;

    if (qr) {
      this.onConnectionUpdateQR(qr);
    }

    if (connection === "open") {
      console.log("✅ Conectado a WhatsApp");
    } else if (connection === "close") {
      this.onConnectionUpdateClose(connection, lastDisconnect);
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

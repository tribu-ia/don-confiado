"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
// you can use this package to export a base64 image or a canvas element.
const qrcode_1 = require("qrcode");
console.log("Hola Mundo");
sock.ev.on('connection.update', (update) => __awaiter(void 0, void 0, void 0, function* () {
    const { connection, lastDisconnect, qr } = update;
    // on a qr event, the connection and lastDisconnect fields will be empty
    // In prod, send this string to your frontend then generate the QR there
    if (qr) {
        // as an example, this prints the qr code to the terminal
        console.log(yield qrcode_1.default.toString(qr, { type: 'terminal' }));
    }
}));

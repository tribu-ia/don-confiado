import { AuthenticationState } from "baileys";

// Tipado correcto del recurso global
interface GlobalResource {
    authState: AuthenticationState | null;
    authSaveCreds: (() => Promise<void>) | null;
}

const globalResource: GlobalResource = {
    authState: null,
    authSaveCreds: null
};

export { globalResource };
export type { GlobalResource };
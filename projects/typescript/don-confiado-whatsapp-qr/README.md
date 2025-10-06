WhatsApp QR service + Simple Chat CLI

Install

```bash
yarn
```

Run WhatsApp QR

```bash
yarn start:wa
```

Run Simple Chat CLI

```bash
# Optional envs:
# BACKEND_URL defaults to http://127.0.0.1:8000/api/chat_v1.0
# USER_ID defaults to cli-user
BACKEND_URL=http://127.0.0.1:8000/api/chat_v1.0 USER_ID=me yarn start:chat
```

You can also run directly:

```bash
yarn tsx src/index.ts chat
```
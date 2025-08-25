import httpx, asyncio, sys, json

async def main():
    ev = {
        "type": "ocr.text",
        "payload": {"text": "This is a test from OCR", "lang": "en"},
        "priority": 5
    }
    async with httpx.AsyncClient() as client:
        r = await client.post("http://127.0.0.1:8765/event", json=ev, timeout=10.0)
        print(r.status_code, r.text)

if __name__ == "__main__":
    asyncio.run(main())

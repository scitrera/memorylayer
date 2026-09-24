"""Optional deployment limits, checked before decoding or GPU work."""

import asyncio
import base64
import io
import json
import os

from starlette.responses import JSONResponse


class RequestLimitsMiddleware:
    def __init__(self, app):
        from ..profiles import service_profile
        self.app = app
        self.profile = service_profile()
        self.enabled = os.environ.get("MEMORYLAYER_EMBED_BOUNDED_PROFILE") == "1"
        self.capacity = int(os.environ.get("MEMORYLAYER_EMBED_MAX_CONCURRENT", "1"))
        self.queue_limit = int(os.environ.get("MEMORYLAYER_EMBED_MAX_QUEUED", "8"))
        self.queue_timeout = float(os.environ.get("MEMORYLAYER_EMBED_QUEUE_TIMEOUT", "600"))
        if self.capacity < 1 or self.queue_limit < 0 or self.queue_timeout <= 0:
            raise ValueError("Invalid request admission limits")
        self.lock = asyncio.Semaphore(self.capacity)
        self.pending = 0
        self.allowed = {"/v1/embeddings", "/v1/embeddings/multi", "/v1/embeddings/images", "/v1/transcribe", "/v1/score"}
        if self.profile == "embedding":
            self.allowed.remove("/v1/transcribe")
            if os.environ.get("MEMORYLAYER_EMBED_GLINER2_ENABLED", "false").lower() == "true":
                self.allowed.add("/v1/ner")
        elif self.profile == "transcription":
            self.allowed = {"/v1/transcribe"}

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"].startswith("/v1/") and self.profile != "combined" and scope["path"] not in self.allowed:
            return await JSONResponse({"detail": "Endpoint disabled by serving profile"}, status_code=404)(scope, receive, send)
        if not self.enabled or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        if scope["path"] not in self.allowed:
            return await JSONResponse({"detail": "Endpoint disabled by serving profile"}, status_code=404)(scope, receive, send)
        # Account before reading bodies: queued callers cannot buffer without bound.
        if self.pending >= self.capacity + self.queue_limit:
            return await JSONResponse({"detail": "GPU queue full"}, status_code=503,
                                      headers={"Retry-After": "1"})(scope, receive, send)
        self.pending += 1
        try:
            await self._bounded(scope, receive, send)
        finally:
            self.pending -= 1

    async def _bounded(self, scope, receive, send):
        body = bytearray()
        try:
            async with asyncio.timeout(30):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        return
                    body.extend(message.get("body", b""))
                    if len(body) > 12 * 1024 * 1024:
                        return await JSONResponse({"detail": "Request exceeds 12 MiB"}, status_code=413)(scope, receive, send)
                    if not message.get("more_body"):
                        break
        except TimeoutError:
            return await JSONResponse({"detail": "Request body timeout"}, status_code=408)(scope, receive, send)
        try:
            data = json.loads(body)
            self.validate(data, text_budget=3072 if scope["path"] == "/v1/embeddings/multi" else 6144)
            if scope["path"] == "/v1/ner":
                texts, labels = data.get("texts", []), data.get("labels", []) or []
                if not isinstance(texts, list) or not 1 <= len(texts) <= 8 or any(not isinstance(t, str) for t in texts):
                    raise ValueError("NER texts")
                if sum(len(t.encode()) for t in texts) > 8192 or len(labels) > 32:
                    raise ValueError("NER budget")
        except (ValueError, TypeError, KeyError, OSError):
            return await JSONResponse({"detail": "Input exceeds serving limits or is invalid"}, status_code=422)(scope, receive, send)
        consumed = False

        async def replay():
            nonlocal consumed
            if not consumed:
                consumed = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=self.queue_timeout)
        except TimeoutError:
            return await JSONResponse({"detail": "GPU busy"}, status_code=503,
                                      headers={"Retry-After": "1"})(scope, receive, send)
        # Keep admission until work really ends even if the external request is
        # cancelled. Cancelling its HTTP await does not cancel native GPU work.
        task = asyncio.create_task(self.app(scope, replay, send))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
            task.result()
            raise
        finally:
            self.lock.release()

    @staticmethod
    def validate(data, *, text_budget=6144):
        if not isinstance(data, dict):
            raise ValueError("object required")
        texts = data.get("input", [])
        texts = [texts] if isinstance(texts, str) else texts
        if not isinstance(texts, list) or len(texts) > 8 or any(not isinstance(t, str) for t in texts):
            raise ValueError("text batch")
        # UTF-8 bytes conservatively bound tokenizer input, including byte fallback.
        if sum(len(t.encode("utf-8")) for t in texts) > text_budget:
            raise ValueError("text budget")
        if "max_tokens" in data and (type(data["max_tokens"]) is not int or not 1 <= data["max_tokens"] <= 4096):
            raise ValueError("generation budget")
        if data.get("dimensions", 1920) != 1920:
            raise ValueError("profile serves 1920 dimensions")
        images = data.get("images", [])
        if not isinstance(images, list) or len(images) > 1:
            raise ValueError("one page per request")
        if images:
            from PIL import Image

            encoded = images[0]
            if not isinstance(encoded, str):
                raise ValueError("base64 image required")
            if encoded.startswith("data:image/"):
                encoded = encoded.partition(",")[2]
            raw = base64.b64decode(encoded, validate=True)
            try:
                with Image.open(io.BytesIO(raw)) as image:
                    w, h = image.size
                    if max(w, h) > 2048 or w * h > 4_000_000 or getattr(image, "n_frames", 1) != 1:
                        raise ValueError("image budget")
                    image.verify()
            except Image.DecompressionBombError as exc:
                raise ValueError("image budget") from exc

"""AWS S3 session store."""

import json
from dataclasses import asdict

from ..session import Session, _parse_messages
from .base import BaseSessionStore


class S3SessionStore(BaseSessionStore):
    def __init__(
        self,
        bucket: str,
        prefix: str = "sessions/",
        region: str = "us-east-1",
    ) -> None:
        import boto3

        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._client = boto3.client("s3", region_name=region)

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}.json"

    def save(self, session: Session) -> None:
        self._client.put_object(
            Bucket=self._bucket,
            Key=self._key(session.id),
            Body=json.dumps(asdict(session), indent=2).encode(),
            ContentType="application/json",
        )

    def load(self, session_id: str) -> Session:
        from botocore.exceptions import ClientError

        try:
            resp = self._client.get_object(
                Bucket=self._bucket,
                Key=self._key(session_id),
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"Session '{session_id}' not found.") from e
            raise
        data = json.loads(resp["Body"].read())
        messages = _parse_messages(data.pop("messages", []))
        return Session(**data, messages=messages)

    def list_all(self) -> list[Session]:
        sessions = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents", []):
                if not obj["Key"].endswith(".json"):
                    continue
                try:
                    resp = self._client.get_object(
                        Bucket=self._bucket, Key=obj["Key"]
                    )
                    data = json.loads(resp["Body"].read())
                    messages = _parse_messages(data.pop("messages", []))
                    sessions.append(Session(**data, messages=messages))
                except Exception:
                    continue
        return sorted(sessions, key=lambda s: s.id)

    def delete(self, session_id: str) -> None:
        self._client.delete_object(
            Bucket=self._bucket,
            Key=self._key(session_id),
        )

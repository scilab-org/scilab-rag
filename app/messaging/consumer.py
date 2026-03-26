"""RabbitMQ consumer for ``PaperIngestionEvent`` messages.

The .NET service publishes messages using **MassTransit** which wraps the
payload in an envelope::

    {
        "messageId": "...",
        "messageType": ["urn:message:..."],
        "message": { <actual payload in camelCase> },
        ...
    }

This consumer unwraps the envelope, runs the KG ingestion pipeline, and
publishes a ``PaperIngestionCompletedEvent`` (plain JSON, since the .NET
side uses ``UseRawJsonDeserializer``).
"""

import json
import logging
from typing import Optional

import aio_pika
from aio_pika.abc import AbstractIncomingMessage

from app.core.config import settings
from app.messaging import connection
from app.messaging.models import PaperIngestionCompletedMessage, PaperIngestionMessage
from app.messaging.publisher import publish_paper_ingestion_completed
from app.services.ingestion_service import ingest_paper_to_kg

logger = logging.getLogger(__name__)


def _unwrap_masstransit_envelope(raw: bytes) -> dict:
    """Extract the inner ``message`` payload from a MassTransit envelope.

    If the body is already a flat dict (no ``message`` key), it is returned
    as-is so that plain-JSON publishers also work during local testing.
    """
    body = json.loads(raw)
    if "message" in body and isinstance(body["message"], dict):
        return body["message"]
    # Fallback: treat the whole body as the payload (useful for testing).
    return body


async def _handle_message(message: AbstractIncomingMessage) -> None:
    """Process a single incoming ``PaperIngestionEvent``."""
    async with message.process():
        logger.info(
            "Received PaperIngestionEvent (delivery_tag=%s)",
            message.delivery_tag,
        )

        try:
            payload = _unwrap_masstransit_envelope(message.body)

            # MassTransit serialises with camelCase; map to our Pydantic model.
            ingestion_msg = PaperIngestionMessage.model_validate(payload)

            logger.info(
                "Processing ingestion for paper %s (%s)",
                ingestion_msg.paper_id,
                ingestion_msg.paper_name,
            )

            result = await ingest_paper_to_kg(
                paper_id=ingestion_msg.paper_id,
                paper_name=ingestion_msg.paper_name,
                parsed_text=ingestion_msg.parsed_text,
            )

            # Publish the outcome back to .NET.
            completed = PaperIngestionCompletedMessage(
                paper_id=ingestion_msg.paper_id,
                is_success=result.success,
                error_message=result.error,
            )
            await publish_paper_ingestion_completed(completed)

        except Exception:
            logger.exception(
                "Unhandled error while processing PaperIngestionEvent "
                "(delivery_tag=%s)",
                message.delivery_tag,
            )
            # Even on unexpected errors, publish a failure event so the .NET
            # side is not left waiting indefinitely.
            try:
                raw_payload = _unwrap_masstransit_envelope(message.body)
                paper_id = raw_payload.get("paperId", raw_payload.get("paper_id"))
                if paper_id is None:
                    raise ValueError("paper_id could not be extracted from payload: {}".format(raw_payload))
                completed = PaperIngestionCompletedMessage(
                    paper_id=str(paper_id),
                    is_success=False,
                    error_message="Unexpected internal error during ingestion",
                )
                await publish_paper_ingestion_completed(completed)
            except Exception:
                logger.exception("Failed to publish failure event")


async def start_consumer() -> Optional[str]:
    """Declare topology and start consuming ``PaperIngestionEvent`` messages.

    Topology mirrors MassTransit defaults:
    - A **fanout** exchange named after the message type.
    - A durable queue bound to that exchange.

    Returns the consumer tag (useful for cancellation on shutdown).
    """
    channel = await connection.get_channel()

    # --- Declare the ingest exchange (fanout, durable) ---
    logger.info(
        "Declaring ingest exchange '%s' (fanout, durable)",
        settings.RABBITMQ_INGEST_EXCHANGE,
    )
    ingest_exchange = await channel.declare_exchange(
        settings.RABBITMQ_INGEST_EXCHANGE,
        type=aio_pika.ExchangeType.FANOUT,
        durable=True,
    )

    # --- Declare the queue ---
    logger.info(
        "Declaring queue '%s' (durable)",
        settings.RABBITMQ_INGEST_QUEUE,
    )
    queue = await channel.declare_queue(
        settings.RABBITMQ_INGEST_QUEUE,
        durable=True,
    )

    # --- Bind queue to exchange ---
    logger.info(
        "Binding queue '%s' to exchange '%s'",
        settings.RABBITMQ_INGEST_QUEUE,
        settings.RABBITMQ_INGEST_EXCHANGE,
    )
    await queue.bind(
        exchange=ingest_exchange,
        routing_key="",
    )
    logger.info(
        "Binding created successfully: queue '%s' -> exchange '%s'",
        settings.RABBITMQ_INGEST_QUEUE,
        settings.RABBITMQ_INGEST_EXCHANGE,
    )

    # --- Also pre-declare the completed exchange so it is ready ---
    await channel.declare_exchange(
        settings.RABBITMQ_COMPLETED_EXCHANGE,
        type=aio_pika.ExchangeType.FANOUT,
        durable=True,
    )

    # --- Start consuming ---
    consumer_tag = await queue.consume(_handle_message)

    logger.info(
        "Consumer started — listening on queue '%s' (exchange '%s')",
        settings.RABBITMQ_INGEST_QUEUE,
        settings.RABBITMQ_INGEST_EXCHANGE,
    )

    return consumer_tag

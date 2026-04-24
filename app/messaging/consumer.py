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
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.db.database import AsyncSessionLocal
from app.db.entities import ProcessedMessage
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
    """Process a single incoming ``PaperIngestionEvent``.

    Design: ack the message immediately after parsing the body, before running
    the 14-minute ingestion pipeline.  This decouples the RabbitMQ ack lifetime
    from the ingestion duration and eliminates ``ChannelInvalidStateError``.

    Previously the code used ``async with message.process()`` which kept the
    ack pending until __aexit__.  On Cloud Run the ingestion takes 10-20 minutes;
    the RabbitMQ channel was closed by the broker before __aexit__ ran, causing:
        aio_pika.message.channel → ChannelInvalidStateError
    """
    logger.info(
        "Received PaperIngestionEvent (delivery_tag=%s)",
        message.delivery_tag,
    )

    # ------------------------------------------------------------------
    # Step 1: Parse the message body.  If parsing fails the message is
    # malformed — nack without requeue so it goes to the dead-letter queue
    # (or is discarded) rather than looping forever.
    # ------------------------------------------------------------------
    try:
        payload = _unwrap_masstransit_envelope(message.body)
        logger.info("Raw unwrapped payload keys: %s", list(payload.keys()))
        logger.debug("Raw unwrapped payload: %s", payload)
        ingestion_msg = PaperIngestionMessage.model_validate(payload)
    except Exception:
        logger.exception(
            "Failed to parse message body (delivery_tag=%s) — nacking without requeue",
            message.delivery_tag,
        )
        await message.nack(requeue=False)
        return

    # ------------------------------------------------------------------
    # Step 2: Ack immediately — ownership is transferred to this service.
    # RabbitMQ is done; the channel state no longer matters for the rest
    # of this function.
    # ------------------------------------------------------------------
    await message.ack()
    logger.info(
        "Message acked (delivery_tag=%s), starting ingestion for paper %s (%s)",
        message.delivery_tag,
        ingestion_msg.paper_id,
        ingestion_msg.paper_name,
    )

    # ------------------------------------------------------------------
    # Step 3: Idempotency guard — claim this paper_id before doing work.
    # A duplicate message (re-sent by .NET after a perceived timeout) hits
    # an IntegrityError here and is silently dropped.
    # ------------------------------------------------------------------
    try:
        async with AsyncSessionLocal() as session:
            try:
                session.add(ProcessedMessage(paper_id=ingestion_msg.paper_id))
                await session.commit()
            except IntegrityError:
                await session.rollback()
                logger.warning(
                    "Duplicate ingestion request for paper %s (%s) — "
                    "already processed, skipping.",
                    ingestion_msg.paper_id,
                    ingestion_msg.paper_name,
                )
                return
    except Exception:
        logger.exception(
            "Failed to write idempotency guard for paper %s — proceeding anyway",
            ingestion_msg.paper_id,
        )

    # ------------------------------------------------------------------
    # Step 4: Run the ingestion pipeline and publish the outcome.
    # ------------------------------------------------------------------
    try:
        result = await ingest_paper_to_kg(
            paper_id=ingestion_msg.paper_id,
            paper_name=ingestion_msg.paper_name,
            parsed_text=ingestion_msg.parsed_text,
            reference_key=ingestion_msg.reference_key,
            authors=ingestion_msg.authors,
            publisher=ingestion_msg.publisher,
            journal_name=ingestion_msg.journal_name,
            volume=ingestion_msg.volume,
            pages=ingestion_msg.pages,
            doi=ingestion_msg.doi,
            publication_month_year=ingestion_msg.publication_month_year,
        )

        if not result.success:
            # Ingestion failed — remove the guard row so a future retry can run.
            try:
                async with AsyncSessionLocal() as session:
                    row = await session.get(ProcessedMessage, ingestion_msg.paper_id)
                    if row is not None:
                        await session.delete(row)
                        await session.commit()
            except Exception:
                logger.exception(
                    "Failed to remove idempotency guard row for paper %s",
                    ingestion_msg.paper_id,
                )

        completed = PaperIngestionCompletedMessage(
            paper_id=ingestion_msg.paper_id,
            is_success=result.success,
            error_message=result.error,
        )
        await publish_paper_ingestion_completed(completed)

    except Exception:
        logger.exception(
            "Unhandled error during ingestion for paper %s (%s)",
            ingestion_msg.paper_id,
            ingestion_msg.paper_name,
        )
        # Remove the guard row so a retry is possible.
        try:
            async with AsyncSessionLocal() as session:
                row = await session.get(ProcessedMessage, ingestion_msg.paper_id)
                if row is not None:
                    await session.delete(row)
                    await session.commit()
        except Exception:
            logger.exception(
                "Failed to remove idempotency guard row for paper %s",
                ingestion_msg.paper_id,
            )
        # Publish a failure event so the .NET side is not left waiting.
        try:
            completed = PaperIngestionCompletedMessage(
                paper_id=ingestion_msg.paper_id,
                is_success=False,
                error_message="Unexpected internal error during ingestion",
            )
            await publish_paper_ingestion_completed(completed)
        except Exception:
            logger.exception(
                "Failed to publish failure event for paper %s",
                ingestion_msg.paper_id,
            )


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

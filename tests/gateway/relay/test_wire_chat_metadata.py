"""Relay wire tests for authoritative chat topic metadata."""

from gateway.relay.ws_transport import _event_from_wire


def _wire_source(**overrides):
    source = {
        "platform": "signal",
        "chat_id": "group:abc",
        "chat_type": "group",
        "chat_name": "House",
        "chat_topic": None,
    }
    source.update(overrides)
    return {
        "text": "hello",
        "message_type": "text",
        "source": source,
    }


def test_authoritative_topic_clear_survives_relay_wire_mapping():
    event = _event_from_wire(_wire_source(chat_topic_known=True))

    assert event.source.chat_topic is None
    assert event.source.chat_topic_known is True


def test_omitted_topic_authority_remains_unknown():
    event = _event_from_wire(_wire_source())

    assert event.source.chat_topic is None
    assert event.source.chat_topic_known is False
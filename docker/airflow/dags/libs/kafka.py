from json import loads
from kafka import KafkaConsumer

def kafka_consumer(server, topic, group_id, consumer_timeout_ms=-1, on_message=None):
    # Create a Kafka Consumer
    consumer = KafkaConsumer(topic,
                             bootstrap_servers=server,
                             auto_offset_reset='earliest',
                             enable_auto_commit=True,
                             group_id=group_id,
                             consumer_timeout_ms=consumer_timeout_ms,
                             value_deserializer=lambda x: loads(x.decode('utf-8')))

    message_head = {'topic': 'Topic name', 'partition': 'Partition', 'offset': 'Offset'}
    print(f"{message_head['topic']:<16} | {message_head['partition']:^10} | {message_head['offset']:<10}")

    messages = []

    # For each new Kafka message
    for message in consumer:
        print(f"\n{message.topic:<16} | {message.partition:^10} | {message.offset:<10}")
        print(f"Body: {message.value}")

        messages.append(message.value)

        if on_message:
            on_message(message)

    consumer.close()

    return messages

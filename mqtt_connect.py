import paho.mqtt.client as mqtt
import time
import sys

# --- Configuration ---
# You can use a free public broker for testing, like Eclipse Mosquitto's test server.
BROKER_ADDRESS = "broker.hivemq.com"
BROKER_PORT = 1883
TOPIC_PATH = "gemini/test/hello" # Use a unique topic path

# --- Callback Functions ---
# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    """
    Called upon successful connection or connection failure.
    rc (return code) 0 means success.
    """
    if rc == 0:
        print(f"Connected to MQTT Broker at {BROKER_ADDRESS}:{BROKER_PORT}!")
    else:
        print(f"Failed to connect, return code {rc}")
        sys.exit(f"Connection error: {rc}")

# The callback for when a PUBLISH message is received from the server (not strictly needed for a publisher, 
# but good practice for handling confirmations, especially with QoS > 0).
def on_publish(client, userdata, mid):
    """
    Called when the message is successfully published.
    mid is the message ID (useful for QoS tracking).
    """
    print(f"Message ID {mid} published successfully.")
    # In a simple script, we can disconnect immediately after publishing.
    client.disconnect()


# --- Main Logic ---
def run_publisher():
    """Sets up the client, connects, and publishes the message."""
    client = mqtt.Client(client_id="PythonPublisherClient")
    
    # 1. Assign callback functions
    client.on_connect = on_connect
    client.on_publish = on_publish

    # 2. Connect to the broker
    try:
        client.connect(BROKER_ADDRESS, BROKER_PORT, keepalive=60)
    except Exception as e:
        print(f"Could not connect to broker: {e}")
        return

    # 3. Start the loop in the background. 
    # This loop handles network traffic, dispatching callbacks, and reconnecting.
    client.loop_start()

    # 4. Wait a moment for the connection to establish before publishing
    time.sleep(1) 

    # 5. Define the payload
    message = "Hello World from Python!"
    
    # 6. Publish the message
    # publish(topic, payload=None, qos=0, retain=False)
    print(f"Attempting to publish '{message}' to topic '{TOPIC_PATH}'...")
    # QoS 1 ensures the message is delivered at least once.
    client.publish(TOPIC_PATH, message, qos=1)

    # 7. Use loop_forever() or a time-based loop to keep the client running 
    # until the on_publish callback calls client.disconnect().
    client.loop_forever() 


if __name__ == '__main__':
    run_publisher()

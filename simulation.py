# simulation_mqtt.py — IoT Fleet Simulator with MQTT Publishing
import pandas as pd
import random
import time
import json
from datetime import datetime
import paho.mqtt.client as mqtt

# --- MQTT Setup ---
BROKER = "broker.hivemq.com"   # public test broker (no auth)
PORT = 1883
TOPIC = "fleet/data"
CLIENT_ID = f"FleetSimulator_{random.randint(1000,9999)}"

client = mqtt.Client(CLIENT_ID)
client.connect(BROKER, PORT, keepalive=60)

# --- Define trucks and fault profiles ---
trucks = {
    "Truck_1": "Normal",
    "Truck_2": "Fuel System Issue",
    "Truck_3": "Coolant Problem",
    "Truck_4": "Oil Pressure Problem",
    "Truck_5": "RPM Fluctuation",
    "Truck_6": "Perfect",
}

# --- Initialize baseline engine state ---
prev_values = {
    truck: {"rpm": random.randint(900, 1100), "temp": random.uniform(75, 85)}
    for truck in trucks
}

# --- Generate engine data per vehicle ---
def generate_engine_data(truck_id, fault_type):
    prev = prev_values[truck_id]
    rpm, temp = prev["rpm"], prev["temp"]

    # Common realistic baseline
    rpm += random.uniform(-50, 50)
    temp += random.uniform(-0.5, 0.5)

    # Fault patterns
    if fault_type == "Fuel System Issue":
        rpm = max(600, rpm - random.uniform(100, 250))
        fuel_pressure = 6 + random.uniform(-1, 0.3)
    else:
        fuel_pressure = 8 + (rpm / 2200) * 10 + random.uniform(-0.5, 0.5)

    if fault_type == "Coolant Problem":
        temp = min(120, temp + random.uniform(5, 10))
        coolant_pressure = 3 + (temp / 120) * 4 + random.uniform(0, 0.3)
    else:
        coolant_pressure = 2 + (temp / 120) * 3 + random.uniform(-0.1, 0.2)

    if fault_type == "Oil Pressure Problem":
        lub_oil_pressure = 1.5 + random.uniform(-0.3, 0.2)
        lub_oil_temp = temp + random.uniform(5, 10)
    else:
        lub_oil_pressure = 2.5 + (rpm / 2200) * 4 + random.uniform(-0.2, 0.2)
        lub_oil_temp = temp + random.uniform(-0.5, 0.5)

    if fault_type == "RPM Fluctuation":
        rpm = max(600, min(2200, rpm + random.uniform(-300, 300)))

    if fault_type == "Perfect":
        rpm = random.uniform(950, 1100)
        temp = random.uniform(75, 85)
        fuel_pressure = 9 + random.uniform(-0.3, 0.3)
        lub_oil_pressure = 3.5 + random.uniform(-0.2, 0.2)
        coolant_pressure = 3 + random.uniform(-0.1, 0.1)
        lub_oil_temp = 80 + random.uniform(-1, 1)
        temp = 80 + random.uniform(-1, 1)

    prev_values[truck_id] = {"rpm": rpm, "temp": temp}

    return {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Vehicle_ID": truck_id,
        "Engine rpm": round(rpm, 2),
        "Lub oil pressure": round(lub_oil_pressure, 2),
        "Fuel pressure": round(fuel_pressure, 2),
        "Coolant pressure": round(coolant_pressure, 2),
        "lub oil temp": round(lub_oil_temp, 2),
        "Coolant temp": round(temp, 2),
    }

# --- Main Simulation Loop ---
def simulate_fleet():
    print(f"Connected to MQTT broker at {BROKER}:{PORT}")
    print("Publishing and saving fleet data every 5 seconds...\n")
    while True:
        data = [generate_engine_data(t, f) for t, f in trucks.items()]
        df = pd.DataFrame(data)
        df.to_csv("fleet_data.csv", index=False)

        for record in data:
            payload = json.dumps(record)
            client.publish(TOPIC, payload)
            print(f"📤 Published to {TOPIC}: {payload}")

        print("\n--- Fleet Snapshot ---")
        print(df)
        time.sleep(5)

if __name__ == "__main__":
    simulate_fleet()

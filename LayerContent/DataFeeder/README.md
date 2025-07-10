# MQTT Data Publisher & Receiver for Badminton 3D Trajectory

This project contains two Python scripts to simulate and observe message exchange over MQTT for 3D shuttlecock trajectory analysis.

## Overview

- `DataFeeder_test.py`: Simulates the publishing of shuttlecock event and segment data by reading from a CSV file.
- `App_test.py`: Subscribes to specific topics and prints out the messages received, acting as a data consumer (e.g., an application or dashboard).

---

## File Structure

```
.
├── App_test.py             # MQTT subscriber for testing received messages
├── DataFeeder_test.py      # MQTT publisher to simulate data stream
├── Content_Output_Sample.csv  # Sample CSV data used by the publisher
```

---

## Requirements

Install required packages:
```bash
pip install paho-mqtt==1.6.1
```

---

## Setup

### Broker

Both scripts are configured to connect to the following broker:

```
Broker IP: 140.113.213.131  
Port: 1886
```

If you need to use a different broker IP or port, modify the initialization line in each script as shown below:

#### In `DataFeeder_test.py`:
```python
app = DataFeeder(broker_ip='your_broker_ip', broker_port='your_broker_port')
```

#### In `App_test.py`:
```python
app = ExampleAPP(broker_ip='your_broker_ip', broker_port='your_broker_port')
```

---

## How to Use

### 1. Start the Receiver

Run `App_test.py` in one terminal:

```bash
python App_test.py
```

It will subscribe to the following MQTT topics and print all received messages:
- `/DATA/ContentDevice/LayerContent/Model3D`
- `/DATA/ContentDevice/LayerContent/Model3D/Event`
- `/DATA/ContentDevice/LayerContent/Model3D/Segment`

---

### 2. Send Simulated Data

In a separate terminal, run `DataFeeder_test.py`:

```bash
python DataFeeder_test.py
```

This script will:
- Read `Content_Output_Sample.csv`
- Publish each row (event or segment) to the MQTT broker
- Delay 0.2 seconds between each publish



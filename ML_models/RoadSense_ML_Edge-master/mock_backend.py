from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/events', methods=['POST'])
def receive_event():
    data = request.get_json()
    print("\n✅ [BACKEND RECEIVED EVENT]")
    print(f"   Type:        {data.get('eventType')}")
    print(f"   Confidence:  {data.get('confidence')}")
    print(f"   Metadata:    {data.get('metadata')}")
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    print("🚀 Mock Backend running on http://127.0.0.1:5000/api/events")
    app.run(port=5000)
